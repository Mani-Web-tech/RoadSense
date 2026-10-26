"""
RoadSense Configuration Module
Centralized configuration parameters, safety thresholds, and model defaults.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
import os

# Supported road user classes (COCO class IDs corresponding to Ultralytics YOLO)
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
COCO_ROAD_USERS: Dict[int, str] = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

@dataclass
class DetectorConfig:
    model_name: str = "yolov8n.pt"  # Lightweight nano model for laptops
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    target_classes: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 5, 7])
    device: str = "cpu"  # 'cpu' or 'cuda' if GPU is available
    imgsz: int = 640

@dataclass
class TrackerConfig:
    tracker_type: str = "bytetrack.yaml"
    track_history_len: int = 25  # Max past trajectory points stored
    min_box_area: float = 400.0  # Filter out noise / minuscule boxes (<20x20)

@dataclass
class TrajectoryConfig:
    prediction_horizon_sec: float = 1.5  # Future prediction window (1.0 to 3.0 sec)
    min_observations: int = 3  # Minimum past points needed before making prediction
    smoothing_factor: float = 0.6  # Exponential moving average factor for velocity
    max_history_seconds: float = 2.0  # Discard observations older than this

@dataclass
class RiskConfig:
    # Explicit Safety Boundary: All metrics are image-space heuristics unless calibrated
    is_calibrated: bool = False

    # Image-space ego corridor: normalized horizontal bounds [x_min, x_max] (0.0 to 1.0)
    # The road lane directly ahead of the vehicle camera
    ego_corridor_left: float = 0.28
    ego_corridor_right: float = 0.72

    # Vertical image threshold: bottom portion represents close proximity
    proximity_y_threshold: float = 0.65  # bottom 35% of frame is near-field

    # Heuristic thresholds
    expansion_rate_medium: float = 0.08   # Bounding box relative growth per second
    expansion_rate_high: float = 0.20     # Rapid closing rate
    heuristic_ttc_high_sec: float = 1.2   # Image-space estimated time-to-contact horizon
    heuristic_ttc_med_sec: float = 2.2

    # Warning state management
    warning_cooldown_sec: float = 2.0     # Suppress repeated alerts for the same event
    persistence_required_frames: int = 2  # Require consecutive frames before elevating to HIGH

@dataclass
class AlertConfig:
    audio_enabled: bool = True
    audio_mode: str = "browser_synth"  # 'browser_synth' (Web Speech), 'tone', 'disabled'
    visual_alerts_enabled: bool = True
    min_alert_level: str = "MEDIUM"     # "LOW", "MEDIUM", "HIGH"

@dataclass
class AppConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)

    # General processing
    display_trails: bool = True
    display_predictions: bool = True
    display_corridor: bool = True
    process_frame_width: int = 720  # Resize input frame for responsive laptop performance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector": {
                "model_name": self.detector.model_name,
                "confidence_threshold": self.detector.confidence_threshold,
                "iou_threshold": self.detector.iou_threshold,
                "device": self.detector.device,
            },
            "trajectory": {
                "prediction_horizon_sec": self.trajectory.prediction_horizon_sec,
                "min_observations": self.trajectory.min_observations,
            },
            "risk": {
                "ego_corridor": [self.risk.ego_corridor_left, self.risk.ego_corridor_right],
                "expansion_rate_high": self.risk.expansion_rate_high,
                "heuristic_ttc_high_sec": self.risk.heuristic_ttc_high_sec,
                "warning_cooldown_sec": self.risk.warning_cooldown_sec,
                "is_calibrated": self.risk.is_calibrated,
            },
            "alerts": {
                "audio_enabled": self.alerts.audio_enabled,
                "audio_mode": self.alerts.audio_mode,
            }
        }
