"""Ghost car: compare two laps of telemetry on a common distance axis (POST /api/ghost/generate).

The CSV files are read and checked here so that mistakes get a precise message before
anything is sent; the API validates the request again and does every calculation.
"""

from __future__ import annotations

import io
import math
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from ui.api_client import ApiError, ApiUnavailable, get_client
from ui.components import heuristic_badge, json_expander, md_text, show_api_error

RESULT_KEY = "ghost_result"
EXAMPLE, UPLOAD = "Example laps", "Upload CSV files"
TRACK_KEYS = {EXAMPLE: "ghost_track_example", UPLOAD: "ghost_track_upload"}

# Mirrors core_modules/ghost_car/schemas.py (the API checks again).
MIN_SAMPLES, MAX_SAMPLES = 2, 20_000
MAX_SPEED_KMH, MAX_GEAR = 400.0, 8
TRACK_SECTION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,39}")
SECONDS, METRES, KMH = (st.column_config.NumberColumn(format=f) for f in ("%.3f", "%.1f", "%.1f"))
# The API's method notes name response fields; they are shown with the page's words for those fields.
METHOD_WORDS = {
    "largest_time_loss/gain.time_s": "The largest loss or gain",
    "delta_time_s": "the gap",
    "from_m/to_m": "its start and end",
    "loss_gain_floor_s": "the resolution",
    "reported as null": "reported as none",
    "brake_point_delta_m": "the brake point delta",
}
SENTENCE_START = r"(?:^|(?<=[.!?] ))"  # a field name that starts a sentence gets a capitalised replacement
ALIGNMENTS = {
    "auto": "Auto: lap fraction for line-to-line laps, otherwise distance",
    "distance": "Distance from each lap's first sample",
    "lap_fraction": "Lap fraction (both traces run timing line to timing line)",
}

TIME_COLUMNS = ("timestamp", "time", "timestamps", "time_s")
SPEED_COLUMNS = ("speed", "speed_kmh")
GEAR_COLUMNS = ("gear", "ngear")  # nGear: FastF1's name
CHANNEL_ALIASES = {"timestamps": TIME_COLUMNS, "speed": SPEED_COLUMNS, "gear": GEAR_COLUMNS}  # lower-case names
OPTIONAL_CHANNELS = ("throttle", "brake", "gear", "drs", "x", "y", "steering")
PEDALS = ("throttle", "brake")
EXTRA_VALUES_PROBLEM = (
    "data rows have more values than the header has column names: check the header row, and that values are separated by commas"
)
BOOLEANS = {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}
CSV_FORMAT = """
| Column | Required | Values |
| --- | --- | --- |
| `timestamp` (or `time`) | yes | seconds, strictly increasing from row to row (any start value) |
| `speed` | yes | km/h, 0-400 |
| `throttle` | no | 0-1 (divide percentages by 100) |
| `brake` | no | 0-1, or true/false for on/off data |
| `gear` (or `nGear`) | no | whole number 0-8 |
| `drs` | no | true/false or 1/0 (flap open) |
| `x`, `y` | no | metres, both or neither; used for distance when both laps have them |
| `steering` | no | any unit; checked and stored, not analysed |

Times are plain seconds, not timedelta text. Every row needs a value in every column it has. Other
columns (such as `distance`) are not used: the API derives distance from speed and time, or from x/y.

FastF1 car data: use `Time.dt.total_seconds()` for the time and divide `Throttle` by 100. Its `DRS`
column holds status codes, not 1/0 (10, 12 and 14 mean open): convert it with `DRS >= 10` first.
"""

# Synthetic example circuit: corner apexes (metres from the line, apex speed in km/h).
EXAMPLE_LAP_M = 5200.0
EXAMPLE_CORNERS = ((650.0, 95.0), (1450.0, 215.0), (2050.0, 130.0), (2900.0, 255.0), (3500.0, 85.0), (4300.0, 165.0))
EXAMPLE_APEX_HALF_WIDTH_M = 20.0
EXAMPLE_TOP_SPEED_KMH = 330.0
EXAMPLE_BRAKING_MS2 = 40.0
EXAMPLE_SAMPLE_HZ = 10.0
EXAMPLE_GEAR_FLOORS_KMH = (80.0, 120.0, 160.0, 200.0, 240.0, 275.0, 300.0)  # upshift speeds into gears 2-8
EXAMPLE_DRS_ZONE_M = (4750.0, 430.0)  # open from 4750 m through the line to 430 m
# Lap 2 against lap 1, per corner (km/h at the apex): slower into turns 1 and 5, quicker through 2 and 6.
EXAMPLE_LAP2_APEX_OFFSETS = (-8.0, 6.0, 0.0, 0.0, -6.0, 4.0)


# ------------------------------------------------------------------ synthetic example laps


def synthetic_lap(apex_offsets: Tuple[float, ...], lap_number: int) -> Dict[str, Any]:
    """One lap of the example circuit as LapTelemetry fields (a kinematic sketch, not recorded data).

    Speed follows the corner limits with a braking and a speed-dependent acceleration envelope
    over two laps, so the second (flying) lap starts and ends at the same speed; it is sampled
    at 10 Hz from line to line, so its timestamps span its lap time exactly.
    """

    step = 1.0
    distance = np.arange(0.0, 2 * EXAMPLE_LAP_M + step, step)
    top = EXAMPLE_TOP_SPEED_KMH / 3.6
    speed = np.full(distance.size, top)
    for (apex, kmh), offset in zip(EXAMPLE_CORNERS, apex_offsets, strict=True):
        for start in (0.0, EXAMPLE_LAP_M):
            zone = np.abs(distance - (start + apex)) <= EXAMPLE_APEX_HALF_WIDTH_M
            speed[zone] = np.minimum(speed[zone], (kmh + offset) / 3.6)
    for i in range(1, distance.size):
        accel = 1.0 + 15.0 * (1.0 - (speed[i - 1] / top) ** 2)  # m/s², falling with speed (drag)
        speed[i] = min(speed[i], math.sqrt(speed[i - 1] ** 2 + 2 * accel * step))
    braking = np.zeros(distance.size, dtype=bool)
    for i in range(distance.size - 2, -1, -1):
        reachable = math.sqrt(speed[i + 1] ** 2 + 2 * EXAMPLE_BRAKING_MS2 * step)
        if reachable < speed[i]:
            speed[i], braking[i] = reachable, True
    flying = distance >= EXAMPLE_LAP_M
    distance, speed, braking = distance[flying] - EXAMPLE_LAP_M, speed[flying], braking[flying]
    elapsed = np.concatenate(([0.0], np.cumsum(2 * step / (speed[1:] + speed[:-1]))))
    lap_time = float(elapsed[-1])

    stamps = np.append(np.arange(0.0, lap_time, 1 / EXAMPLE_SAMPLE_HZ), lap_time)
    position = np.interp(stamps, elapsed, distance)
    kmh = np.interp(stamps, elapsed, speed) * 3.6
    brake = np.interp(position, distance, braking.astype(float)) >= 0.5
    at_apex = np.zeros(stamps.size, dtype=bool)
    for apex, _ in EXAMPLE_CORNERS:
        at_apex |= np.abs(position - apex) <= EXAMPLE_APEX_HALF_WIDTH_M
    start_drs, end_drs = EXAMPLE_DRS_ZONE_M
    splits = np.interp([EXAMPLE_LAP_M / 3, 2 * EXAMPLE_LAP_M / 3], distance, elapsed)
    return {
        "lap_number": lap_number,
        "timestamps": np.round(stamps, 3).tolist(),
        "speed": np.round(kmh, 1).tolist(),
        "throttle": np.where(brake, 0.0, np.where(at_apex, 0.35, 1.0)).tolist(),
        "brake": brake.astype(float).tolist(),
        "gear": (1 + np.searchsorted(EXAMPLE_GEAR_FLOORS_KMH, kmh, side="right")).astype(int).tolist(),
        "drs": ((position >= start_drs) | (position < end_drs)).tolist(),
        "lap_time": round(lap_time, 3),
        "sector_times": [
            round(float(splits[0]), 3),
            round(float(splits[1] - splits[0]), 3),
            round(lap_time - float(splits[1]), 3),
        ],
    }


@st.cache_data(show_spinner=False)
def example_laps() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return synthetic_lap((0.0,) * len(EXAMPLE_CORNERS), 14), synthetic_lap(EXAMPLE_LAP2_APEX_OFFSETS, 15)


@st.cache_data(show_spinner=False)
def template_csv() -> bytes:
    """The example reference lap in the upload format."""

    lap = example_laps()[0]
    columns = {"timestamp": lap["timestamps"], "speed": lap["speed"]}
    columns.update({name: lap[name] for name in ("throttle", "brake", "gear", "drs")})
    return pd.DataFrame(columns).to_csv(index=False).encode()


# ------------------------------------------------------------------ CSV reading


def quoted(name: Any) -> str:
    """A file or column name as a Markdown code span."""

    return "`" + str(name).replace("`", "'") + "`"


def parse_cell(text: str, channel: str) -> Any:
    """One CSV value of ``channel``; raises ValueError with the reason."""

    lowered = text.lower()
    if channel == "drs" or (channel == "brake" and lowered in ("true", "false")):
        if lowered not in BOOLEANS:
            hint = " (FastF1 DRS codes: convert with DRS >= 10)" if channel == "drs" and lowered.isdigit() else ""
            raise ValueError(f"{text!r} is not true/false or 1/0{hint}")
        return BOOLEANS[lowered] if channel == "drs" else float(BOOLEANS[lowered])
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"{text!r} is not a number") from None
    if not math.isfinite(number):
        raise ValueError(f"{text!r} is not a finite number")
    if channel in PEDALS and not 0.0 <= number <= 1.0:
        raise ValueError(f"{number:g} is outside 0-1 (divide percentages by 100)")
    if channel == "speed" and not 0.0 <= number <= MAX_SPEED_KMH:
        raise ValueError(f"{number:g} is outside 0-{MAX_SPEED_KMH:.0f} km/h")
    if channel == "gear":
        if not number.is_integer() or not 0 <= number <= MAX_GEAR:
            raise ValueError(f"{text!r} is not a gear number 0-{MAX_GEAR}")
        return int(number)
    return number


def read_lap_csv(data: bytes) -> Tuple[Dict[str, List[Any]], List[str], List[str]]:
    """``(channels, notes, problems)`` for one lap CSV; channels are empty when there are problems."""

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", pd.errors.ParserWarning)
            # index_col=False: rows with more values than the header must not turn the first column into an index
            frame = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, encoding="utf-8-sig", index_col=False)
    except UnicodeDecodeError:
        return {}, [], ["the file is not UTF-8 text; export it as CSV (UTF-8)"]
    except pd.errors.EmptyDataError:
        return {}, [], ["the file is empty"]
    except pd.errors.ParserError as exc:
        return {}, [], [f"the file is not a valid CSV ({str(exc).strip().splitlines()[-1]})"]
    if any(issubclass(warning.category, pd.errors.ParserWarning) for warning in caught):
        return {}, [], [EXTRA_VALUES_PROBLEM]

    columns: Dict[str, str] = {}  # channel -> column name in the file
    problems: List[str] = []
    for raw in frame.columns:
        name = str(raw).strip().lower()
        channel = next((channel for channel, names in CHANNEL_ALIASES.items() if name in names), name)
        if channel in columns:
            problems.append(f"columns {quoted(columns[channel])} and {quoted(raw)} both hold {channel}; keep one")
        columns[channel] = str(raw)
    found = ", ".join(quoted(name) for name in frame.columns) or "none"
    if len(frame.columns) == 1 and ";" in str(frame.columns[0]):
        found += "; the file seems to use ';' between values: save it with commas"
    for channel, names in (("timestamps", TIME_COLUMNS), ("speed", SPEED_COLUMNS)):
        if channel not in columns:
            problems.append(f"missing the {quoted(names[0])} column (columns found: {found})")
    ignored = [raw for channel, raw in columns.items() if channel not in ("timestamps", "speed", *OPTIONAL_CHANNELS)]
    notes = [f"ignored columns: {', '.join(map(quoted, ignored))} (not used by the comparison)"] if ignored else []
    if not MIN_SAMPLES <= len(frame) <= MAX_SAMPLES:
        problems.append(f"it has {len(frame)} data rows; {MIN_SAMPLES} to {MAX_SAMPLES} are needed")
    if problems:
        return {}, notes, problems

    channels: Dict[str, List[Any]] = {}
    for channel, raw in columns.items():
        if raw in ignored:
            continue
        values: List[Any] = []
        for row, cell in enumerate(frame[raw].tolist(), start=1):
            text = cell.strip() if isinstance(cell, str) else ""  # short rows give NaN
            if not text:
                problems.append(f"column {quoted(raw)}: data row {row} is empty (fill it, or remove the column)")
                break
            try:
                values.append(parse_cell(text, channel))
            except ValueError as exc:
                problems.append(f"column {quoted(raw)}: data row {row}: {exc}")
                break
        channels[channel] = values
    stamps = channels.get("timestamps", [])
    if len(stamps) == len(frame):
        for row in range(1, len(stamps)):
            if stamps[row] <= stamps[row - 1]:
                problems.append(
                    f"time must increase from row to row: data row {row + 1} ({stamps[row]:g} s) "
                    f"does not come after data row {row} ({stamps[row - 1]:g} s)"
                )
                break
    return ({} if problems else channels), notes, problems


def parse_sector_times(text: str) -> Optional[List[float]]:
    """``"28.1, 38.4, 27.0"`` -> three sector times; empty -> None; raises ValueError."""

    if not text.strip():
        return None
    try:
        values = [float(part) for part in re.split(r"[,;\s]+", text.strip())]
    except ValueError:
        raise ValueError("sector times must be numbers in seconds, e.g. 28.1, 38.4, 27.0") from None
    if len(values) != 3 or not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("give exactly three positive sector times in seconds")
    return values


def uploaded_lap(key: str, label: str) -> Tuple[Optional[Dict[str, Any]], List[str], List[str]]:
    """``(telemetry, notes, problems)`` for one uploaded lap and its optional timing fields."""

    upload = st.session_state.get(f"ghost_{key}_file")
    if upload is None:
        return None, [], [f"{label}: upload a CSV file"]
    channels, notes, problems = read_lap_csv(upload.getvalue())
    notes = [f"{label} ({quoted(upload.name)}): {note}" for note in notes]
    problems = [f"{label} ({quoted(upload.name)}): {problem}" for problem in problems]
    telemetry: Dict[str, Any] = dict(channels)
    for field in ("lap_number", "lap_time"):
        value = st.session_state.get(f"ghost_{key}_{field}")
        if value is not None:
            telemetry[field] = value
    try:
        sectors = parse_sector_times(st.session_state.get(f"ghost_{key}_sectors", ""))
    except ValueError as exc:
        problems.append(f"{label}: {exc}")
    else:
        if sectors is not None:
            telemetry["sector_times"] = sectors
    return (None if problems else telemetry), notes, problems


# ------------------------------------------------------------------ result


def method_text(text: str) -> str:
    """An API method note as Markdown, with plain words for the response fields it names."""

    for field, words in METHOD_WORDS.items():
        text = re.sub(SENTENCE_START + re.escape(field), words[:1].upper() + words[1:], text).replace(field, words)
    return md_text(text)


def shown_delta(value: float, decimals: int) -> float:
    """A signed difference rounded as displayed, so a tiny negative value shows as 0.0, not -0.0."""

    return round(value, decimals) + 0.0


def zone_text(zone: Optional[Dict[str, Any]], floor: Any) -> str:
    if zone is None:
        return f"none above the {floor} s resolution"
    return f"{zone['time_s']:.3f} s between {zone['from_m']:.0f} m and {zone['to_m']:.0f} m"


def lap_rows(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for key, role in (("lap1", "Reference (lap 1)"), ("lap2", "Comparison (lap 2)")):
        lap = body["laps"][key]
        rows.append(
            {
                "Lap": role,
                "Lap number": lap.get("lap_number"),
                "Samples": lap.get("samples"),
                "Telemetry span (s)": lap.get("telemetry_duration_s"),
                "Distance (m)": lap.get("distance_m"),
                "Lap time (s)": lap.get("lap_time_s"),
                "Lap time from": str(lap.get("lap_time_source", "")).replace("_", " "),
                "Missing channels": ", ".join(body["missing_channels"][key]) or "none",
            }
        )
    return rows


def render_channel_tables(body: Dict[str, Any]) -> None:
    sectors, braking = body.get("sector_time_deltas_s"), body.get("braking")
    tabs = st.tabs(["Laps", "Sectors", "Braking", "Throttle, gears, DRS", "Aligned traces"])
    with tabs[0]:
        st.dataframe(
            lap_rows(body),
            hide_index=True,
            column_config={"Telemetry span (s)": SECONDS, "Distance (m)": METRES, "Lap time (s)": SECONDS},
        )
    with tabs[1]:
        if sectors is None:
            st.caption("Sector deltas need sector_times for both laps.")
        else:
            laps = body["laps"]
            st.dataframe(
                [
                    {
                        "Sector": f"S{number}",
                        "Lap 1 (s)": laps["lap1"]["sector_times_s"][number - 1],
                        "Lap 2 (s)": laps["lap2"]["sector_times_s"][number - 1],
                        "Lap 2 - lap 1 (s)": shown_delta(delta, 3),
                    }
                    for number, delta in enumerate(sectors, start=1)
                ],
                hide_index=True,
                column_config={name: SECONDS for name in ("Lap 1 (s)", "Lap 2 (s)", "Lap 2 - lap 1 (s)")},
            )
    with tabs[2]:
        if braking is None:
            st.caption("Braking zones need the brake channel in both laps.")
        else:
            st.caption(method_text(braking["heuristic"]))
            st.dataframe(
                [
                    {
                        "Lap 1 braking from (m)": zone["lap1_start_m"],
                        "Lap 2 braking from (m)": zone["lap2_start_m"],
                        "Brake point delta (m; > 0 = lap 2 later)": shown_delta(zone["brake_point_delta_m"], 1),
                        "Lap 1 braking to (m)": zone["lap1_end_m"],
                        "Lap 2 braking to (m)": zone["lap2_end_m"],
                    }
                    for zone in braking["matched_zones"]
                ],
                hide_index=True,
                column_config={
                    name: METRES
                    for name in (
                        "Lap 1 braking from (m)",
                        "Lap 2 braking from (m)",
                        "Brake point delta (m; > 0 = lap 2 later)",
                        "Lap 1 braking to (m)",
                        "Lap 2 braking to (m)",
                    )
                },
            )
            st.caption(
                f"Braking zones: {braking['lap1_zone_count']} in lap 1, {braking['lap2_zone_count']} in lap 2, "
                f"{braking['matched_zone_count']} matched."
            )
            if braking.get("zones_truncated"):
                st.caption("Only the first zones are listed.")
    with tabs[3]:
        rows = []
        throttle, gear, drs = body.get("throttle"), body.get("gear"), body.get("drs")
        if throttle:
            shares = (throttle["lap1_full_throttle_fraction"], throttle["lap2_full_throttle_fraction"])
            rows.append(("Full-throttle share of distance", *(f"{share:.1%}" for share in shares)))
        if gear:
            rows.append(("Highest gear", str(gear["lap1_max_gear"]), str(gear["lap2_max_gear"])))
            rows.append(("Gear changes", str(gear["lap1_gear_changes"]), str(gear["lap2_gear_changes"])))
        if drs:
            rows.append(("DRS open distance (m)", f"{drs['lap1_open_distance_m']:.0f}", f"{drs['lap2_open_distance_m']:.0f}"))
        if rows:
            st.dataframe([{"Measure": name, "Lap 1": one, "Lap 2": two} for name, one, two in rows], hide_index=True)
            if throttle:
                st.caption(md_text(throttle["heuristic"]))
        else:
            st.caption("Throttle, gear and DRS comparisons need those channels in both laps.")
    with tabs[4]:
        traces = body["traces"]
        st.dataframe(
            pd.DataFrame(
                {
                    "Distance (m)": traces["distance_m"],
                    "Gap lap 2 - lap 1 (s)": [shown_delta(value, 3) for value in traces["delta_time_s"]],
                    "Lap 1 speed (km/h)": traces["lap1_speed_kmh"],
                    "Lap 2 speed (km/h)": traces["lap2_speed_kmh"],
                    "Speed delta (km/h)": [shown_delta(value, 1) for value in traces["speed_delta_kmh"]],
                }
            ),
            hide_index=True,
            column_config={
                "Distance (m)": METRES,
                "Gap lap 2 - lap 1 (s)": SECONDS,
                "Lap 1 speed (km/h)": KMH,
                "Lap 2 speed (km/h)": KMH,
                "Speed delta (km/h)": KMH,
            },
        )


def render_result(result: Dict[str, Any]) -> None:
    body = result["response"]
    summary, alignment = body["summary"], body["alignment"]
    st.subheader("Comparison")
    st.caption(f"Compared: {result['label']}. Gaps are lap 2 minus lap 1 at the same position: above 0 means lap 2 is behind.")
    columns = st.columns(4)
    columns[0].metric(
        "Gap at the end",
        f"{shown_delta(summary['final_delta_s'], 3):+.3f} s",
        help=f"Lap 2 minus lap 1 at {summary['final_delta_at_m']:.0f} m, the end of the compared distance",
    )
    lap_delta = body.get("lap_time_delta_s")
    columns[1].metric(
        "Official lap-time delta",
        "not supplied" if lap_delta is None else f"{shown_delta(lap_delta, 3):+.3f} s",
        help="Lap 2 minus lap 1 from lap_time or the sector times; needs them for both laps (lap_time_delta_s in "
        "the API response and its warnings)",
    )
    floor = summary.get("loss_gain_floor_s")
    loss, gain = summary.get("largest_time_loss"), summary.get("largest_time_gain")
    for column, zone, label in ((columns[2], loss, "Largest loss (lap 2)"), (columns[3], gain, "Largest gain (lap 2)")):
        column.metric(label, "none" if zone is None else f"{zone['time_s']:.3f} s", help="Heuristic location; see below")
    st.caption(
        f"Largest loss: {zone_text(loss, floor)}; largest gain: {zone_text(gain, floor)}. Each is the largest net change "
        "of the gap over any stretch, so a stretch can contain shorter gains and losses (see the chart). "
        f"Method: {method_text(summary.get('loss_gain_heuristic', ''))}"
    )
    st.caption(
        f"Largest gap {shown_delta(summary['max_delta_s'], 3):+.3f} s at {summary['max_delta_at_m']:.0f} m, smallest "
        f"{shown_delta(summary['min_delta_s'], 3):+.3f} s at {summary['min_delta_at_m']:.0f} m. Alignment: "
        f"**{alignment['method'].replace('_', ' ')}** (requested {alignment['requested'].replace('_', ' ')}), distance from "
        f"{alignment['distance_source'].replace('_', ' ')} over {alignment['compared_distance_m']:.0f} m. "
        f"{md_text(alignment['description'])}"
    )
    for warning in body.get("warnings") or []:
        st.warning(md_text(warning), icon=":material/warning:")
    if result["image"] is not None:
        st.image(result["image"], caption="Speed and gap along the lap, rendered by the API", width="stretch")
    else:
        show_api_error(result["image_error"], "Fetching the comparison image")
    render_channel_tables(body)
    json_expander(body)


# ------------------------------------------------------------------ page


st.title("Ghost car")
heuristic_badge("Heuristic zones · not validated")
st.markdown(
    "Compare a **reference lap** (lap 1) with a **comparison lap** (lap 2). The API puts both laps on a "
    "common distance axis (speed integrated over time, or x/y when both laps have it) and reports how far "
    "lap 2 is ahead or behind at every point. Channels a lap does not supply are reported, never filled in. "
    "Braking-zone matching and the loss/gain locations are heuristic."
)

source = st.segmented_control("Telemetry", [EXAMPLE, UPLOAD], default=EXAMPLE, required=True, key="ghost_source")
if source == EXAMPLE:
    st.info(
        "**Synthetic example data, not recorded telemetry.** This page generates two 10 Hz laps of a made-up "
        "5.2 km circuit from a simple speed model (lap 14 and lap 15; lap 15 is slower through turns 1 and 5 "
        "and quicker through turns 2 and 6). Use it to see what the comparison shows.",
        icon=":material/science:",
    )
else:
    with st.expander("CSV format", icon=":material/table:"):
        st.markdown(CSV_FORMAT)
    st.download_button(
        "Download a CSV template",
        data=template_csv(),
        file_name="ghost_lap_template.csv",
        mime="text/csv",
        on_click="ignore",
        icon=":material/download:",
        help="The synthetic example reference lap in the upload format",
    )

with st.form("ghost_form"):
    if source == UPLOAD:
        for column, key, role in zip(st.columns(2), ("lap1", "lap2"), ("Reference", "Comparison"), strict=True):
            with column:
                st.markdown(f"**{role} lap ({key.replace('lap', 'lap ')})**")
                st.file_uploader(f"{role} lap CSV", type=["csv"], key=f"ghost_{key}_file")
                st.number_input(
                    f"{role} lap number (optional)",
                    min_value=1,
                    max_value=200,
                    value=None,
                    step=1,
                    key=f"ghost_{key}_lap_number",
                )
                st.number_input(
                    f"{role} lap time in s (optional)",
                    min_value=0.001,
                    max_value=3600.0,
                    value=None,
                    format="%.3f",
                    key=f"ghost_{key}_lap_time",
                    help="From timing, not the telemetry. Enables the lap-time delta and, when the samples span it, "
                    "lap-fraction alignment.",
                )
                st.text_input(
                    f"{role} sector times in s (optional)", placeholder="28.103, 38.412, 27.015", key=f"ghost_{key}_sectors"
                )
    left, right = st.columns(2)
    left.text_input(
        "Track or section name (optional)",
        value="Example circuit" if source == EXAMPLE else "",
        max_chars=40,
        key=TRACK_KEYS[source],
        help="Letters, digits, spaces, _ and -; starts with a letter or digit. Used as the chart title.",
    )
    right.selectbox("Alignment", list(ALIGNMENTS), format_func=ALIGNMENTS.get, key="ghost_alignment")
    submitted = st.form_submit_button("Compare laps", type="primary", icon=":material/compare_arrows:", key="ghost_submit")

if submitted:
    st.session_state.pop(RESULT_KEY, None)
    problems: List[str] = []
    track_section = st.session_state.get(TRACK_KEYS[source], "").strip()
    if track_section and not TRACK_SECTION_PATTERN.fullmatch(track_section):
        problems.append("Track or section name: use letters, digits, spaces, _ and -, starting with a letter or digit")
    if source == EXAMPLE:
        lap1, lap2 = example_laps()
        label = "the synthetic example laps 14 and 15 (not recorded telemetry)"
    else:
        lap1, notes1, problems1 = uploaded_lap("lap1", "Reference lap")
        lap2, notes2, problems2 = uploaded_lap("lap2", "Comparison lap")
        problems += problems1 + problems2
        for note in notes1 + notes2:
            st.info(note, icon=":material/info:")
        names = [
            st.session_state[f"ghost_{key}_file"].name for key in ("lap1", "lap2") if st.session_state.get(f"ghost_{key}_file")
        ]
        label = " (reference) vs ".join(map(quoted, names)) + " (comparison)"
    if problems:
        st.error(
            "Nothing was sent to the API. Please fix:\n" + "\n".join(f"- {problem}" for problem in problems),
            icon=":material/rule:",
        )
    else:
        payload: Dict[str, Any] = {
            "lap1_telemetry": lap1,
            "lap2_telemetry": lap2,
            "alignment": st.session_state["ghost_alignment"],
        }
        if track_section:
            payload["track_section"] = track_section
        client = get_client()
        try:
            with st.spinner("Comparing the laps..."):
                body = client.generate_ghost(payload)
        except (ApiUnavailable, ApiError) as exc:
            show_api_error(exc, "The comparison")
            if isinstance(exc, ApiError) and exc.status_code == 422 and source == UPLOAD:
                st.caption(
                    "In these messages `lap1_telemetry` is the reference CSV and `lap2_telemetry` the comparison CSV; "
                    "`[i]` is data row i + 1 (sample i counts from 0)."
                )
        else:
            image: Optional[bytes] = None
            image_error: Optional[Exception] = None
            try:
                image = client.fetch_ghost_image(body["visualization_url"])
            except (ApiUnavailable, ApiError) as exc:
                image_error = exc
            st.session_state[RESULT_KEY] = {"label": label, "response": body, "image": image, "image_error": image_error}

if RESULT_KEY in st.session_state:
    render_result(st.session_state[RESULT_KEY])
