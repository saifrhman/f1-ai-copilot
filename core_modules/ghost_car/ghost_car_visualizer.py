#!/usr/bin/env python3
"""Ghost-car telemetry comparison and headless visualization generation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


@dataclass
class TelemetryPoint:
    timestamp: float
    x: float
    y: float
    speed: float
    throttle: float
    brake: float
    steering: float
    drs: bool
    gear: int


@dataclass
class LapData:
    lap_number: int
    driver: str
    telemetry_points: List[TelemetryPoint]
    lap_time: float
    sector_times: List[float]


class GhostCarVisualizer:
    """Compare two laps from supplied telemetry and write visual artifacts under outputs/."""

    def __init__(self, output_dir: str = "outputs/ghost"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.track_layouts = {
            "monaco": {"name": "Circuit de Monaco", "length": 3337, "corners": 19},
            "silverstone": {"name": "Silverstone Circuit", "length": 5891, "corners": 18},
            "spa": {"name": "Circuit de Spa-Francorchamps", "length": 7004, "corners": 20},
        }

    def generate_ghost_comparison(
        self,
        lap1_telemetry: Dict[str, Any],
        lap2_telemetry: Dict[str, Any],
        track_section: str = "monaco",
    ) -> Dict[str, Any]:
        lap1 = self._parse_telemetry(lap1_telemetry, "Lap 1")
        lap2 = self._parse_telemetry(lap2_telemetry, "Lap 2")
        artifact = self._generate_visualization(lap1, lap2, track_section)
        return {
            "frame_deltas": self._calculate_frame_deltas(lap1, lap2),
            "speed_comparison": self._compare_speeds(lap1, lap2),
            "braking_zones": self._analyze_braking_zones(lap1, lap2),
            "drs_usage": self._analyze_drs_usage(lap1, lap2),
            "visualization_url": artifact["url"],
            "visualization_path": artifact["path"],
            "track_info": self.track_layouts.get(track_section, {"name": track_section}),
            "lap_times": {
                "lap1": lap1.lap_time,
                "lap2": lap2.lap_time,
                "delta": float(lap2.lap_time - lap1.lap_time),
            },
            "sector_times": {"lap1": lap1.sector_times, "lap2": lap2.sector_times},
            "samples_compared": min(len(lap1.telemetry_points), len(lap2.telemetry_points)),
        }

    @staticmethod
    def _series(telemetry: Dict[str, Any], name: str, n: int, default: Any) -> List[Any]:
        values = telemetry.get(name)
        if values is None:
            return [default(i) if callable(default) else default for i in range(n)]
        if not isinstance(values, list):
            raise ValueError(f"{name} must be a list")
        if len(values) != n:
            raise ValueError(f"{name} must contain exactly {n} samples")
        return values

    def _parse_telemetry(self, telemetry: Dict[str, Any], lap_name: str) -> LapData:
        timestamps = telemetry.get("timestamps")
        if not isinstance(timestamps, list) or not timestamps:
            raise ValueError("timestamps must be a non-empty list")
        n = len(timestamps)
        timestamps = [float(v) for v in timestamps]
        if any(not np.isfinite(v) for v in timestamps):
            raise ValueError("timestamps must be finite")
        if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError("timestamps must be strictly increasing")

        speeds = [float(v) for v in self._series(telemetry, "speed", n, 0.0)]
        x_coords = [float(v) for v in self._series(telemetry, "x", n, lambda i: float(i))]
        y_coords = [float(v) for v in self._series(telemetry, "y", n, 0.0)]
        throttles = [float(v) for v in self._series(telemetry, "throttle", n, 0.0)]
        brakes = [float(v) for v in self._series(telemetry, "brake", n, 0.0)]
        steerings = [float(v) for v in self._series(telemetry, "steering", n, 0.0)]
        drs = [bool(v) for v in self._series(telemetry, "drs", n, False)]
        gears = [int(v) for v in self._series(telemetry, "gear", n, 1)]

        numeric_series = speeds + x_coords + y_coords + throttles + brakes + steerings
        if any(not np.isfinite(v) for v in numeric_series):
            raise ValueError("telemetry contains non-finite numeric values")

        inferred_lap_time = timestamps[-1] - timestamps[0]
        lap_time = float(telemetry.get("lap_time", inferred_lap_time))
        if lap_time <= 0:
            raise ValueError("lap_time must be positive")
        sector_times = telemetry.get("sector_times")
        if sector_times is None:
            sector_times = [lap_time / 3.0] * 3
        if not isinstance(sector_times, list) or len(sector_times) != 3:
            raise ValueError("sector_times must contain exactly three values")
        sector_times = [float(v) for v in sector_times]

        points = [
            TelemetryPoint(
                timestamp=timestamps[i],
                x=x_coords[i],
                y=y_coords[i],
                speed=speeds[i],
                throttle=throttles[i],
                brake=brakes[i],
                steering=steerings[i],
                drs=drs[i],
                gear=gears[i],
            )
            for i in range(n)
        ]
        return LapData(
            lap_number=int(telemetry.get("lap_number", 1)),
            driver=lap_name,
            telemetry_points=points,
            lap_time=lap_time,
            sector_times=sector_times,
        )

    @staticmethod
    def _calculate_frame_deltas(lap1: LapData, lap2: LapData) -> List[float]:
        n = min(len(lap1.telemetry_points), len(lap2.telemetry_points))
        start1 = lap1.telemetry_points[0].timestamp
        start2 = lap2.telemetry_points[0].timestamp
        return [
            float((lap2.telemetry_points[i].timestamp - start2) - (lap1.telemetry_points[i].timestamp - start1))
            for i in range(n)
        ]

    @staticmethod
    def _compare_speeds(lap1: LapData, lap2: LapData) -> Dict[str, List[float]]:
        n = min(len(lap1.telemetry_points), len(lap2.telemetry_points))
        s1 = [float(p.speed) for p in lap1.telemetry_points[:n]]
        s2 = [float(p.speed) for p in lap2.telemetry_points[:n]]
        return {"lap1_speeds": s1, "lap2_speeds": s2, "speed_deltas": [b - a for a, b in zip(s1, s2)]}

    @staticmethod
    def _analyze_braking_zones(lap1: LapData, lap2: LapData) -> Dict[str, List[int]]:
        b1 = [i for i, p in enumerate(lap1.telemetry_points) if p.brake > 0.1]
        b2 = [i for i, p in enumerate(lap2.telemetry_points) if p.brake > 0.1]
        return {"lap1_braking_zones": b1, "lap2_braking_zones": b2, "braking_differences": sorted(set(b1) ^ set(b2))}

    @staticmethod
    def _analyze_drs_usage(lap1: LapData, lap2: LapData) -> Dict[str, List[bool]]:
        n = min(len(lap1.telemetry_points), len(lap2.telemetry_points))
        d1 = [p.drs for p in lap1.telemetry_points[:n]]
        d2 = [p.drs for p in lap2.telemetry_points[:n]]
        return {"lap1_drs_usage": d1, "lap2_drs_usage": d2, "drs_differences": [a != b for a, b in zip(d1, d2)]}

    def _generate_visualization(self, lap1: LapData, lap2: LapData, track_section: str) -> Dict[str, str]:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8))
        self._plot_track_overlay(axes[0], lap1, lap2, track_section)
        self._plot_speed_comparison(axes[1], lap1, lap2)
        fig.tight_layout()

        filename = f"ghost_comparison_{track_section}_{lap1.lap_number}_{lap2.lap_number}.png"
        path = self.output_dir / filename
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        try:
            relative = path.relative_to(Path("outputs"))
            url = "/artifacts/" + relative.as_posix()
        except ValueError:
            url = path.as_posix()
        return {"path": str(path), "url": url}

    def _plot_track_overlay(self, ax: Any, lap1: LapData, lap2: LapData, track_section: str) -> None:
        x1 = [p.x for p in lap1.telemetry_points]
        y1 = [p.y for p in lap1.telemetry_points]
        x2 = [p.x for p in lap2.telemetry_points]
        y2 = [p.y for p in lap2.telemetry_points]
        track_name = self.track_layouts.get(track_section, {}).get("name", track_section)
        ax.plot(x1, y1, linewidth=1.8, label=f"Lap {lap1.lap_number}")
        ax.plot(x2, y2, linewidth=1.8, label=f"Lap {lap2.lap_number}")
        ax.scatter([x1[0]], [y1[0]], s=35, label="Start")
        ax.set_title(f"Telemetry path comparison - {track_name}")
        ax.set_xlabel("X / sample-progress coordinate")
        ax.set_ylabel("Y coordinate")
        ax.legend()
        ax.grid(True, alpha=0.25)

    @staticmethod
    def _plot_speed_comparison(ax: Any, lap1: LapData, lap2: LapData) -> None:
        t1 = [p.timestamp - lap1.telemetry_points[0].timestamp for p in lap1.telemetry_points]
        t2 = [p.timestamp - lap2.telemetry_points[0].timestamp for p in lap2.telemetry_points]
        ax.plot(t1, [p.speed for p in lap1.telemetry_points], linewidth=1.8, label=f"Lap {lap1.lap_number}")
        ax.plot(t2, [p.speed for p in lap2.telemetry_points], linewidth=1.8, label=f"Lap {lap2.lap_number}")
        ax.set_title("Speed trace")
        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel("Speed (km/h)")
        ax.legend()
        ax.grid(True, alpha=0.25)

    def create_animation(self, lap1: LapData, lap2: LapData, track_section: str) -> str:
        fig, ax = plt.subplots(figsize=(10, 7))
        n = min(len(lap1.telemetry_points), len(lap2.telemetry_points))

        def animate(frame: int) -> None:
            ax.clear()
            p1 = lap1.telemetry_points[: frame + 1]
            p2 = lap2.telemetry_points[: frame + 1]
            ax.plot([p.x for p in p1], [p.y for p in p1], linewidth=1.8, label=f"Lap {lap1.lap_number}")
            ax.plot([p.x for p in p2], [p.y for p in p2], linewidth=1.8, label=f"Lap {lap2.lap_number}")
            ax.scatter([p1[-1].x, p2[-1].x], [p1[-1].y, p2[-1].y], s=35)
            ax.set_title(f"Ghost comparison - frame {frame}")
            ax.legend()

        animation = FuncAnimation(fig, animate, frames=n, interval=50, repeat=False)
        filename = f"ghost_animation_{track_section}_{lap1.lap_number}_{lap2.lap_number}.gif"
        path = self.output_dir / filename
        animation.save(path, writer="pillow")
        plt.close(fig)
        return str(path)


_visualizer: Optional[GhostCarVisualizer] = None


def get_visualizer() -> GhostCarVisualizer:
    global _visualizer
    if _visualizer is None:
        _visualizer = GhostCarVisualizer()
    return _visualizer


def generate_ghost_comparison(
    lap1_telemetry: Dict[str, Any],
    lap2_telemetry: Dict[str, Any],
    track_section: str = "monaco",
) -> Dict[str, Any]:
    return get_visualizer().generate_ghost_comparison(lap1_telemetry, lap2_telemetry, track_section)
