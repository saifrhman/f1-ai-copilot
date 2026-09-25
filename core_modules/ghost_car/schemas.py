"""Validated request models for the ghost-car lap comparison.

These models are used directly as FastAPI request bodies. Every numeric
channel is strict (``bool``, ``None`` and strings are rejected, never coerced)
and finite, so nothing downstream has to guess what a value meant.
"""

from enum import Enum
from typing import Annotated, List, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

MIN_SAMPLES = 2
MAX_SAMPLES = 20_000
MAX_SPEED_KMH = 400.0
MAX_LAP_TIME_S = 3_600.0
TRACK_SECTION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,39}$"

# Sector splits are timed to the millisecond, so their sum may differ from the
# lap time by a few ms of rounding; anything beyond this is inconsistent input.
SECTOR_SUM_TOLERANCE_S = 0.05
# Telemetry may carry a sample or two either side of the timing line, so its
# time span may exceed the lap time slightly, but never by more than this.
SPAN_TOLERANCE_S = 0.5
SPAN_TOLERANCE_FRACTION = 0.02
# A lap counts as running "line to line" (first and last sample on the timing
# line) only when its timestamps span its official lap time within this; the
# 'auto' alignment uses lap_fraction only for such laps.
LINE_TO_LINE_TOLERANCE_S = 0.01
AUTO_LAP_FRACTION_TOLERANCE = 0.03

ALL_OPTIONAL_CHANNELS = ("x", "y", "brake", "throttle", "steering", "drs", "gear")


def _finite(**bounds: float) -> object:
    return Field(strict=True, allow_inf_nan=False, **bounds)


Timestamp = Annotated[float, _finite(ge=-1e10, le=1e10)]
Coordinate = Annotated[float, _finite(ge=-1e6, le=1e6)]
SpeedKmh = Annotated[float, _finite(ge=0.0, le=MAX_SPEED_KMH)]
Pedal = Annotated[float, _finite(ge=0.0, le=1.0)]
Steering = Annotated[float, _finite(ge=-1e6, le=1e6)]
Gear = Annotated[int, Field(strict=True, ge=0, le=8)]
Seconds = Annotated[float, _finite(gt=0.0, le=MAX_LAP_TIME_S)]


def _channel(description: str, required: bool = False) -> object:
    default = ... if required else None
    return Field(default, min_length=MIN_SAMPLES, max_length=MAX_SAMPLES, description=description)


class AlignmentMode(str, Enum):
    """How the two laps are put on a common axis (see ``ghost_car_visualizer``)."""

    AUTO = "auto"
    LAP_FRACTION = "lap_fraction"
    DISTANCE = "distance"


class LapTelemetry(BaseModel):
    """Per-sample telemetry of one lap (or one section of a lap).

    ``timestamps`` and ``speed`` are required. Every other channel is optional,
    but when supplied it must have exactly one value per timestamp. Missing
    channels are reported by the comparison; they are never filled in.
    """

    model_config = ConfigDict(extra="forbid")

    lap_number: Optional[int] = Field(
        None, strict=True, ge=1, le=200, description="Lap number, used only for labels."
    )
    timestamps: List[Timestamp] = _channel(
        "Sample times in seconds, strictly increasing (any origin).", required=True
    )
    speed: List[SpeedKmh] = _channel("Speed in km/h (0-400).", required=True)
    x: Optional[List[Coordinate]] = _channel("Car X position in metres; requires y.")
    y: Optional[List[Coordinate]] = _channel("Car Y position in metres; requires x.")
    brake: Optional[List[Pedal]] = _channel("Brake application 0..1 (0/1 for on/off data).")
    throttle: Optional[List[Pedal]] = _channel("Throttle application 0..1.")
    steering: Optional[List[Steering]] = _channel("Steering input; validated and stored, not analysed.")
    drs: Optional[List[StrictBool]] = _channel("DRS flap open (true/false).")
    gear: Optional[List[Gear]] = _channel("Selected gear, integer 0 (neutral) to 8.")
    lap_time: Optional[Seconds] = Field(
        None,
        description=(
            "Official timing-line lap time in seconds. When the timestamps span it within "
            f"{LINE_TO_LINE_TOLERANCE_S} s the trace is treated as running from line to line."
        ),
    )
    sector_times: Optional[List[Seconds]] = Field(
        None, min_length=3, max_length=3, description="The three official sector times in seconds."
    )

    @model_validator(mode="after")
    def _check_consistency(self) -> "LapTelemetry":
        stamps = self.timestamps
        for index in range(1, len(stamps)):
            if stamps[index] <= stamps[index - 1]:
                raise ValueError(
                    f"timestamps must be strictly increasing (sample {index}: "
                    f"{stamps[index]} follows {stamps[index - 1]})"
                )
        span = stamps[-1] - stamps[0]
        if span > MAX_LAP_TIME_S:
            raise ValueError(
                f"timestamps span {span:.1f} s, more than the {MAX_LAP_TIME_S:.0f} s allowed for one lap; "
                "timestamps must be in seconds (not milliseconds)"
            )
        n = len(stamps)
        for name in ("speed",) + ALL_OPTIONAL_CHANNELS:
            values = getattr(self, name)
            if values is not None and len(values) != n:
                raise ValueError(f"{name} has {len(values)} samples but timestamps has {n}")
        if (self.x is None) != (self.y is None):
            raise ValueError("x and y must be supplied together")
        self._check_timing()
        return self

    def _check_timing(self) -> None:
        if self.sector_times is not None and sum(self.sector_times) > MAX_LAP_TIME_S:
            raise ValueError(
                f"sector_times sum to {sum(self.sector_times):.3f} s, more than the "
                f"{MAX_LAP_TIME_S:.0f} s allowed for one lap"
            )
        if self.sector_times is not None and self.lap_time is not None:
            total = sum(self.sector_times)
            if abs(total - self.lap_time) > SECTOR_SUM_TOLERANCE_S:
                raise ValueError(
                    f"sector_times sum to {total:.3f} s but lap_time is {self.lap_time:.3f} s"
                )
        official = self.lap_time if self.lap_time is not None else (
            sum(self.sector_times) if self.sector_times is not None else None
        )
        if official is not None:
            span = self.timestamps[-1] - self.timestamps[0]
            allowed = official + max(SPAN_TOLERANCE_S, SPAN_TOLERANCE_FRACTION * official)
            if span > allowed:
                raise ValueError(
                    f"telemetry spans {span:.3f} s, longer than the lap time of {official:.3f} s"
                )

    def missing_channels(self) -> List[str]:
        """Optional channels this lap did not supply, in a fixed order."""

        return [name for name in ALL_OPTIONAL_CHANNELS if getattr(self, name) is None]


class GhostCarRequest(BaseModel):
    """Compare a reference lap (lap 1) with a comparison lap (lap 2)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    lap1_telemetry: LapTelemetry = Field(description="Reference lap; deltas are lap 2 minus lap 1.")
    lap2_telemetry: LapTelemetry = Field(description="Comparison lap.")
    track_section: Optional[str] = Field(
        None,
        pattern=TRACK_SECTION_PATTERN,
        description="Label for the track or section (letters, digits, space, _ and -; max 40).",
    )
    alignment: AlignmentMode = Field(
        AlignmentMode.AUTO,
        description=(
            "'distance' compares equal metres from each lap's first sample over the shared "
            "distance range (assumes the same start point); 'lap_fraction' maps both laps onto "
            "0..1 of their own distance (assumes both traces run from timing line to timing line); "
            "'auto' uses lap_fraction only when both laps' timestamps span their official lap time "
            f"(lap_time or sector_times) within {LINE_TO_LINE_TOLERANCE_S} s and their total "
            f"distances agree within {AUTO_LAP_FRACTION_TOLERANCE:.0%}, otherwise distance."
        ),
    )
