#!/usr/bin/env python3
"""
Ghost Car Visualizer
Renders overlay visualizations from telemetry data for lap comparisons
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation


@dataclass
class TelemetryPoint:
    """Single telemetry data point"""
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
    """Complete lap telemetry data"""
    lap_number: int
    driver: str
    telemetry_points: List[TelemetryPoint]
    lap_time: float
    sector_times: List[float]


class GhostCarVisualizer:
    """Creates ghost car visualizations for lap comparisons"""
    
    def __init__(self):
        self.track_layouts = self._load_track_layouts()
        self.colors = {
            'lap1': '#FF6B6B',  # Red
            'lap2': '#4ECDC4',  # Teal
            'ghost': '#95A5A6',  # Gray
            'track': '#2C3E50',  # Dark blue
            'background': '#ECF0F1'  # Light gray
        }
    
    def _load_track_layouts(self) -> Dict[str, Dict[str, Any]]:
        """Load track layout data"""
        return {
            "monaco": {
                "name": "Circuit de Monaco",
                "length": 3337,  # meters
                "corners": 19,
                "sectors": [
                    {"start": 0, "end": 0.33, "name": "Sector 1"},
                    {"start": 0.33, "end": 0.67, "name": "Sector 2"},
                    {"start": 0.67, "end": 1.0, "name": "Sector 3"}
                ]
            },
            "silverstone": {
                "name": "Silverstone Circuit",
                "length": 5891,
                "corners": 18,
                "sectors": [
                    {"start": 0, "end": 0.33, "name": "Sector 1"},
                    {"start": 0.33, "end": 0.67, "name": "Sector 2"},
                    {"start": 0.67, "end": 1.0, "name": "Sector 3"}
                ]
            },
            "spa": {
                "name": "Circuit de Spa-Francorchamps",
                "length": 7004,
                "corners": 20,
                "sectors": [
                    {"start": 0, "end": 0.33, "name": "Sector 1"},
                    {"start": 0.33, "end": 0.67, "name": "Sector 2"},
                    {"start": 0.67, "end": 1.0, "name": "Sector 3"}
                ]
            }
        }
    
    def generate_ghost_comparison(
        self,
        lap1_telemetry: Dict[str, Any],
        lap2_telemetry: Dict[str, Any],
        track_section: str = "monaco"
    ) -> Dict[str, Any]:
        """
        Generate ghost car comparison visualization
        
        Args:
            lap1_telemetry: Telemetry data for first lap
            lap2_telemetry: Telemetry data for second lap
            track_section: Track section for comparison
            
        Returns:
            Comparison data with deltas and visualization info
        """
        # Parse telemetry data
        lap1_data = self._parse_telemetry(lap1_telemetry, "Lap 1")
        lap2_data = self._parse_telemetry(lap2_telemetry, "Lap 2")
        
        # Calculate frame-by-frame deltas
        frame_deltas = self._calculate_frame_deltas(lap1_data, lap2_data)
        
        # Generate speed comparison
        speed_comparison = self._compare_speeds(lap1_data, lap2_data)
        
        # Analyze braking zones
        braking_zones = self._analyze_braking_zones(lap1_data, lap2_data)
        
        # Analyze DRS usage
        drs_usage = self._analyze_drs_usage(lap1_data, lap2_data)
        
        # Generate visualization
        visualization_data = self._generate_visualization(
            lap1_data, lap2_data, track_section
        )
        
        return {
            "frame_deltas": frame_deltas,
            "speed_comparison": speed_comparison,
            "braking_zones": braking_zones,
            "drs_usage": drs_usage,
            "visualization_url": visualization_data.get("url"),
            "track_info": self.track_layouts.get(track_section, {}),
            "lap_times": {
                "lap1": lap1_data.lap_time,
                "lap2": lap2_data.lap_time,
                "delta": lap2_data.lap_time - lap1_data.lap_time
            },
            "sector_times": {
                "lap1": lap1_data.sector_times,
                "lap2": lap2_data.sector_times
            }
        }
    
    def _parse_telemetry(self, telemetry: Dict[str, Any], lap_name: str) -> LapData:
        """Parse telemetry data into structured format"""
        points = []
        
        # Extract telemetry points
        timestamps = telemetry.get("timestamps", [])
        x_coords = telemetry.get("x", [])
        y_coords = telemetry.get("y", [])
        speeds = telemetry.get("speed", [])
        throttles = telemetry.get("throttle", [])
        brakes = telemetry.get("brake", [])
        steerings = telemetry.get("steering", [])
        drs_states = telemetry.get("drs", [])
        gears = telemetry.get("gear", [])
        
        # Create telemetry points
        for i in range(len(timestamps)):
            point = TelemetryPoint(
                timestamp=timestamps[i] if i < len(timestamps) else i * 0.1,
                x=x_coords[i] if i < len(x_coords) else 0,
                y=y_coords[i] if i < len(y_coords) else 0,
                speed=speeds[i] if i < len(speeds) else 0,
                throttle=throttles[i] if i < len(throttles) else 0,
                brake=brakes[i] if i < len(brakes) else 0,
                steering=steerings[i] if i < len(steerings) else 0,
                drs=drs_states[i] if i < len(drs_states) else False,
                gear=gears[i] if i < len(gears) else 1
            )
            points.append(point)
        
        # Calculate lap time and sector times
        lap_time = telemetry.get("lap_time", len(points) * 0.1)
        sector_times = telemetry.get("sector_times", [lap_time/3, lap_time/3, lap_time/3])
        
        return LapData(
            lap_number=telemetry.get("lap_number", 1),
            driver=lap_name,
            telemetry_points=points,
            lap_time=lap_time,
            sector_times=sector_times
        )
    
    def _calculate_frame_deltas(self, lap1: LapData, lap2: LapData) -> List[float]:
        """Calculate frame-by-frame time deltas"""
        deltas = []
        
        # Interpolate to same time points
        min_length = min(len(lap1.telemetry_points), len(lap2.telemetry_points))
        
        for i in range(min_length):
            # Calculate time delta at this frame
            time1 = lap1.telemetry_points[i].timestamp
            time2 = lap2.telemetry_points[i].timestamp
            delta = time2 - time1
            deltas.append(delta)
        
        return deltas
    
    def _compare_speeds(self, lap1: LapData, lap2: LapData) -> Dict[str, List[float]]:
        """Compare speed traces between laps"""
        speeds1 = [point.speed for point in lap1.telemetry_points]
        speeds2 = [point.speed for point in lap2.telemetry_points]
        
        # Pad shorter array
        max_length = max(len(speeds1), len(speeds2))
        speeds1.extend([speeds1[-1]] * (max_length - len(speeds1)))
        speeds2.extend([speeds2[-1]] * (max_length - len(speeds2)))
        
        return {
            "lap1_speeds": speeds1,
            "lap2_speeds": speeds2,
            "speed_deltas": [s2 - s1 for s1, s2 in zip(speeds1, speeds2)]
        }
    
    def _analyze_braking_zones(self, lap1: LapData, lap2: LapData) -> Dict[str, List[int]]:
        """Analyze braking zones comparison"""
        braking1 = [i for i, point in enumerate(lap1.telemetry_points) if point.brake > 0.1]
        braking2 = [i for i, point in enumerate(lap2.telemetry_points) if point.brake > 0.1]
        
        return {
            "lap1_braking_zones": braking1,
            "lap2_braking_zones": braking2,
            "braking_differences": list(set(braking1) ^ set(braking2))  # XOR
        }
    
    def _analyze_drs_usage(self, lap1: LapData, lap2: LapData) -> Dict[str, List[bool]]:
        """Analyze DRS usage comparison"""
        drs1 = [point.drs for point in lap1.telemetry_points]
        drs2 = [point.drs for point in lap2.telemetry_points]
        
        # Pad shorter array
        max_length = max(len(drs1), len(drs2))
        drs1.extend([drs1[-1]] * (max_length - len(drs1)))
        drs2.extend([drs2[-1]] * (max_length - len(drs2)))
        
        return {
            "lap1_drs_usage": drs1,
            "lap2_drs_usage": drs2,
            "drs_differences": [d1 != d2 for d1, d2 in zip(drs1, drs2)]
        }
    
    def _generate_visualization(
        self,
        lap1: LapData,
        lap2: LapData,
        track_section: str
    ) -> Dict[str, Any]:
        """Generate visualization data"""
        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Track overlay
        self._plot_track_overlay(ax1, lap1, lap2, track_section)
        
        # Plot 2: Speed comparison
        self._plot_speed_comparison(ax2, lap1, lap2)
        
        # Save visualization
        filename = f"ghost_comparison_{track_section}_{lap1.lap_number}_{lap2.lap_number}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            "url": filename,
            "track_section": track_section,
            "lap_numbers": [lap1.lap_number, lap2.lap_number]
        }
    
    def _plot_track_overlay(self, ax, lap1: LapData, lap2: LapData, track_section: str):
        """Plot track overlay with both laps"""
        # Extract coordinates
        x1 = [point.x for point in lap1.telemetry_points]
        y1 = [point.y for point in lap1.telemetry_points]
        x2 = [point.x for point in lap2.telemetry_points]
        y2 = [point.y for point in lap2.telemetry_points]
        
        # Plot track outline (simplified)
        track_info = self.track_layouts.get(track_section, {})
        track_length = track_info.get("length", 5000)
        
        # Create simple track outline
        track_x = np.linspace(0, track_length, 100)
        track_y = 50 * np.sin(track_x / 1000)  # Simple sinusoidal track
        
        ax.plot(track_x, track_y, 'k-', linewidth=3, alpha=0.3, label='Track')
        
        # Plot lap traces
        ax.plot(x1, y1, color=self.colors['lap1'], linewidth=2, label=f'{lap1.driver} ({lap1.lap_time:.1f}s)')
        ax.plot(x2, y2, color=self.colors['lap2'], linewidth=2, label=f'{lap2.driver} ({lap2.lap_time:.1f}s)')
        
        # Mark start/finish
        ax.plot(x1[0], y1[0], 'go', markersize=10, label='Start/Finish')
        
        ax.set_title(f'Ghost Car Comparison - {track_info.get("name", track_section)}')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_speed_comparison(self, ax, lap1: LapData, lap2: LapData):
        """Plot speed comparison"""
        timestamps1 = [point.timestamp for point in lap1.telemetry_points]
        speeds1 = [point.speed for point in lap1.telemetry_points]
        timestamps2 = [point.timestamp for point in lap2.telemetry_points]
        speeds2 = [point.speed for point in lap2.telemetry_points]
        
        ax.plot(timestamps1, speeds1, color=self.colors['lap1'], linewidth=2, label=f'{lap1.driver}')
        ax.plot(timestamps2, speeds2, color=self.colors['lap2'], linewidth=2, label=f'{lap2.driver}')
        
        # Highlight speed differences
        min_length = min(len(timestamps1), len(timestamps2))
        for i in range(min_length):
            if abs(speeds1[i] - speeds2[i]) > 10:  # Significant speed difference
                ax.axvspan(timestamps1[i], timestamps1[i] + 0.1, alpha=0.2, color='yellow')
        
        ax.set_title('Speed Comparison')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (km/h)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def create_animation(self, lap1: LapData, lap2: LapData, track_section: str) -> str:
        """Create animated ghost car visualization"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Setup track
        track_info = self.track_layouts.get(track_section, {})
        track_length = track_info.get("length", 5000)
        track_x = np.linspace(0, track_length, 100)
        track_y = 50 * np.sin(track_x / 1000)
        
        ax.plot(track_x, track_y, 'k-', linewidth=3, alpha=0.3)
        ax.set_xlim(0, track_length)
        ax.set_ylim(-100, 100)
        ax.set_title(f'Ghost Car Animation - {track_info.get("name", track_section)}')
        
        # Animation function
        def animate(frame):
            ax.clear()
            ax.plot(track_x, track_y, 'k-', linewidth=3, alpha=0.3)
            
            # Plot cars up to current frame
            if frame < len(lap1.telemetry_points):
                x1 = [point.x for point in lap1.telemetry_points[:frame+1]]
                y1 = [point.y for point in lap1.telemetry_points[:frame+1]]
                ax.plot(x1, y1, color=self.colors['lap1'], linewidth=2)
                ax.plot(x1[-1], y1[-1], 'o', color=self.colors['lap1'], markersize=8)
            
            if frame < len(lap2.telemetry_points):
                x2 = [point.x for point in lap2.telemetry_points[:frame+1]]
                y2 = [point.y for point in lap2.telemetry_points[:frame+1]]
                ax.plot(x2, y2, color=self.colors['lap2'], linewidth=2)
                ax.plot(x2[-1], y2[-1], 'o', color=self.colors['lap2'], markersize=8)
            
            ax.set_xlim(0, track_length)
            ax.set_ylim(-100, 100)
            ax.set_title(f'Ghost Car Animation - Frame {frame}')
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=min(len(lap1.telemetry_points), len(lap2.telemetry_points)), 
                           interval=50, repeat=False)
        
        # Save animation
        filename = f"ghost_animation_{track_section}_{lap1.lap_number}_{lap2.lap_number}.gif"
        anim.save(filename, writer='pillow')
        plt.close()
        
        return filename


# Global visualizer instance
_visualizer = None

def get_visualizer() -> GhostCarVisualizer:
    """Get or create visualizer instance"""
    global _visualizer
    if _visualizer is None:
        _visualizer = GhostCarVisualizer()
    return _visualizer

def generate_ghost_comparison(
    lap1_telemetry: Dict[str, Any],
    lap2_telemetry: Dict[str, Any],
    track_section: str = "monaco"
) -> Dict[str, Any]:
    """
    Generate ghost car comparison
    
    Args:
        lap1_telemetry: Telemetry data for first lap
        lap2_telemetry: Telemetry data for second lap
        track_section: Track section for comparison
        
    Returns:
        Comparison data with deltas and visualization info
    """
    visualizer = get_visualizer()
    return visualizer.generate_ghost_comparison(lap1_telemetry, lap2_telemetry, track_section)


# Example usage and testing
if __name__ == "__main__":
    print("🏁 Ghost Car Visualizer Test")
    print("=" * 50)
    
    # Create mock telemetry data
    mock_lap1 = {
        "timestamps": list(range(100)),
        "x": [i * 10 for i in range(100)],
        "y": [50 * np.sin(i/10) for i in range(100)],
        "speed": [200 + 20 * np.sin(i/10) for i in range(100)],
        "throttle": [0.8 + 0.2 * np.sin(i/5) for i in range(100)],
        "brake": [0.1 if i % 20 < 5 else 0 for i in range(100)],
        "steering": [0.1 * np.sin(i/5) for i in range(100)],
        "drs": [i % 30 > 20 for i in range(100)],
        "gear": [6 for _ in range(100)],
        "lap_time": 85.2,
        "sector_times": [28.1, 28.5, 28.6],
        "lap_number": 14
    }
    
    mock_lap2 = {
        "timestamps": list(range(100)),
        "x": [i * 10 + 5 for i in range(100)],
        "y": [50 * np.sin(i/10 + 0.1) for i in range(100)],
        "speed": [200 + 25 * np.sin(i/10) for i in range(100)],
        "throttle": [0.85 + 0.15 * np.sin(i/5) for i in range(100)],
        "brake": [0.15 if i % 20 < 4 else 0 for i in range(100)],
        "steering": [0.12 * np.sin(i/5) for i in range(100)],
        "drs": [i % 30 > 18 for i in range(100)],
        "gear": [6 for _ in range(100)],
        "lap_time": 84.8,
        "sector_times": [27.9, 28.3, 28.6],
        "lap_number": 15
    }
    
    visualizer = get_visualizer()
    
    # Generate comparison
    result = visualizer.generate_ghost_comparison(mock_lap1, mock_lap2, "monaco")
    
    print(f"📊 Comparison Results:")
    print(f"   • Lap Times: {result['lap_times']['lap1']:.1f}s vs {result['lap_times']['lap2']:.1f}s")
    print(f"   • Delta: {result['lap_times']['delta']:.1f}s")
    print(f"   • Frame Deltas: {len(result['frame_deltas'])} frames analyzed")
    print(f"   • Braking Zones: {len(result['braking_zones']['lap1_braking_zones'])} vs {len(result['braking_zones']['lap2_braking_zones'])}")
    print(f"   • DRS Usage: {sum(result['drs_usage']['lap1_drs_usage'])} vs {sum(result['drs_usage']['lap2_drs_usage'])} activations")
    print(f"   • Visualization: {result['visualization_url']}")
    
    # Test animation
    print(f"\n🎬 Creating animation...")
    animation_file = visualizer.create_animation(
        visualizer._parse_telemetry(mock_lap1, "Lap 1"),
        visualizer._parse_telemetry(mock_lap2, "Lap 2"),
        "monaco"
    )
    print(f"   • Animation saved: {animation_file}") 