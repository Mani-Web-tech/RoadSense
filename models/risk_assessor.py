"""
RoadSense Collision-Risk Assessment Module
Evaluates image-space conflict potential between detected road users and the camera vehicle.

SAFETY NOTICE:
All calculations are uncalibrated image-space heuristics based on pixel kinematics.
Without camera extrinsics (height, pitch), focal length, and vehicle speed CAN-telemetry,
real-world 3D distance and exact physical time-to-collision cannot be determined.
Risk levels are categorized as SAFE, LOW RISK, MEDIUM RISK, HIGH RISK, or INSUFFICIENT DATA.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np

from config import RiskConfig
from models.tracker import TrackedObject
from models.trajectory_predictor import TrajectoryPrediction
from utils.logging_config import logger


class RiskLevel:
    SAFE = "SAFE"
    LOW = "LOW RISK"
    MEDIUM = "MEDIUM RISK"
    HIGH = "HIGH RISK"
    INSUFFICIENT_DATA = "INSUFFICIENT DATA"
    NO_TARGETS_DETECTED = "NO TARGETS DETECTED"


@dataclass
class ObjectRiskAssessment:
    """Individual collision-risk assessment for a single tracked road user."""
    track_id: int
    risk_level: str
    risk_score: float                              # Heuristic index in [0.0, 1.0]
    reason: str
    uncalibrated_ttc_sec: Optional[float]          # Image-space kinematic heuristic time in seconds (or None)
    is_in_ego_corridor: bool
    predicted_corridor_entry: bool
    object_class: str
    confidence: float
    stability_score: float
    warning_triggered: bool = False                # True only if a new alert should fire (subject to cooldown)
    calibration_status: str = "UNCALIBRATED_IMAGE_SPACE_HEURISTIC"


@dataclass
class SceneRiskSummary:
    """Consolidated risk status for the overall scene at the current frame timestamp."""
    overall_risk_level: str
    highest_risk_object_id: Optional[int]
    highest_risk_reason: str
    active_objects_count: int
    assessments: List[ObjectRiskAssessment] = field(default_factory=list)
    timestamp: float = 0.0
    scene_reliability: str = "NORMAL"              # "NORMAL", "LOW_STABILITY", "EMPTY"


class RiskAssessor:
    """
    Evaluates conflict potential and manages alert state transitions and cooldowns.
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        # Track last alert dispatch timestamp per track_id to prevent alarm floods
        self._last_alert_timestamps: Dict[int, float] = {}
        self._global_last_alert_timestamp: float = -999.0

    def reset(self) -> None:
        """Resets risk evaluation cache."""
        self._last_alert_timestamps.clear()
        self._global_last_alert_timestamp = -999.0

    def assess_object(
        self,
        track_obj: TrackedObject,
        prediction: TrajectoryPrediction,
        frame_width: int,
        frame_height: int,
        current_timestamp: float
    ) -> ObjectRiskAssessment:
        """
        Assesses conflict risk for an individual road user.
        """
        cx, cy = track_obj.center
        corridor_left = frame_width * self.config.ego_corridor_left
        corridor_right = frame_width * self.config.ego_corridor_right
        proximity_y = frame_height * self.config.proximity_y_threshold

        is_in_corridor = corridor_left <= cx <= corridor_right
        is_close_proximity = cy >= proximity_y

        # Check if future predicted path intersects or enters the ego corridor
        predicted_corridor_entry = False
        if prediction.is_reliable and prediction.predicted_points:
            for px, py in prediction.predicted_points:
                if corridor_left <= px <= corridor_right and py >= (frame_height * 0.4):
                    predicted_corridor_entry = True
                    break

        # Check track stability
        if track_obj.stability_score < 0.4 or not prediction.is_reliable:
            return ObjectRiskAssessment(
                track_id=track_obj.track_id,
                risk_level=RiskLevel.INSUFFICIENT_DATA,
                risk_score=0.1,
                reason="Track observation period too brief or unstable for confident risk estimation",
                uncalibrated_ttc_sec=None,
                is_in_ego_corridor=is_in_corridor,
                predicted_corridor_entry=predicted_corridor_entry,
                object_class=track_obj.class_name,
                confidence=track_obj.confidence,
                stability_score=track_obj.stability_score,
                warning_triggered=False
            )

        # Calculate image-space heuristic time-to-contact (TTC)
        vx, vy = prediction.velocity_xy
        uncalibrated_ttc: Optional[float] = None
        # In camera coordinates, positive vy indicates moving down towards the vehicle
        if vy > 15.0 and cy < frame_height:
            # Uncalibrated image-space projection: pixels remaining / closing velocity
            uncalibrated_ttc = round((frame_height - cy) / vy, 2)
            if uncalibrated_ttc < 0:
                uncalibrated_ttc = None

        is_vulnerable = track_obj.class_name in ("pedestrian", "bicycle")
        is_approaching = (
            prediction.motion_direction == "Approaching" or
            prediction.area_growth_rate >= self.config.expansion_rate_medium or
            (uncalibrated_ttc is not None and uncalibrated_ttc < 4.0)
        )
        is_rapid_closing = (
            prediction.area_growth_rate >= self.config.expansion_rate_high or
            (uncalibrated_ttc is not None and uncalibrated_ttc <= self.config.heuristic_ttc_high_sec)
        )

        # Risk Decision Logic
        level = RiskLevel.SAFE
        score = 0.0
        reason = "Object outside vehicle path with no projected conflict."

        # Scenario 1: Vulnerable road user in corridor or entering corridor
        if is_vulnerable:
            if is_in_corridor or (predicted_corridor_entry and is_close_proximity):
                level = RiskLevel.HIGH
                score = 0.95
                reason = f"Vulnerable road user ({track_obj.class_name}) directly inside vehicle forward corridor."
            elif predicted_corridor_entry:
                level = RiskLevel.MEDIUM
                score = 0.70
                reason = f"Vulnerable road user ({track_obj.class_name}) projected to enter forward corridor."
            else:
                level = RiskLevel.LOW
                score = 0.35
                reason = f"Vulnerable road user ({track_obj.class_name}) present in peripheral zone."

        # Scenario 2: Motor vehicle in corridor
        elif is_in_corridor:
            if is_rapid_closing or (uncalibrated_ttc is not None and uncalibrated_ttc <= self.config.heuristic_ttc_high_sec):
                level = RiskLevel.HIGH
                score = 0.90
                ttc_str = f" [Image-space closing metric: ~{uncalibrated_ttc}s (uncalibrated)]" if uncalibrated_ttc else ""
                reason = f"{track_obj.class_name.capitalize()} closing rapidly ahead in lane corridor{ttc_str}."
            elif is_approaching or is_close_proximity:
                level = RiskLevel.MEDIUM
                score = 0.65
                ttc_str = f" [Image-space closing metric: ~{uncalibrated_ttc}s (uncalibrated)]" if uncalibrated_ttc else ""
                reason = f"{track_obj.class_name.capitalize()} ahead in forward corridor with active approach{ttc_str}."
            elif prediction.motion_direction == "Receding":
                level = RiskLevel.SAFE
                score = 0.10
                reason = f"{track_obj.class_name.capitalize()} moving away (receding) ahead."
            else:
                level = RiskLevel.LOW
                score = 0.30
                reason = f"{track_obj.class_name.capitalize()} present in forward corridor at stable distance."

        # Scenario 3: Object entering corridor laterally
        elif predicted_corridor_entry:
            if is_approaching or is_close_proximity:
                level = RiskLevel.MEDIUM
                score = 0.60
                reason = f"{track_obj.class_name.capitalize()} predicted to cross into forward vehicle corridor."
            else:
                level = RiskLevel.LOW
                score = 0.35
                reason = f"{track_obj.class_name.capitalize()} lateral motion approaching corridor boundary."

        # Scenario 4: Peripheral / safe movement
        else:
            if prediction.motion_direction == "Stationary":
                level = RiskLevel.SAFE
                score = 0.05
                reason = f"Stationary {track_obj.class_name} outside travel lane."
            elif prediction.motion_direction == "Receding":
                level = RiskLevel.SAFE
                score = 0.02
                reason = f"{track_obj.class_name.capitalize()} moving away outside vehicle corridor."
            else:
                level = RiskLevel.LOW
                score = 0.20
                reason = f"{track_obj.class_name.capitalize()} moving in adjacent lane with non-conflicting trajectory."

        # Determine if audio/prominent warning should fire (enforce cooldown)
        warning_triggered = False
        if level in (RiskLevel.HIGH, RiskLevel.MEDIUM):
            last_t = self._last_alert_timestamps.get(track_obj.track_id, -999.0)
            if (current_timestamp - last_t) >= self.config.warning_cooldown_sec:
                warning_triggered = True
                self._last_alert_timestamps[track_obj.track_id] = current_timestamp
                self._global_last_alert_timestamp = current_timestamp

        return ObjectRiskAssessment(
            track_id=track_obj.track_id,
            risk_level=level,
            risk_score=score,
            reason=reason,
            uncalibrated_ttc_sec=uncalibrated_ttc,
            is_in_ego_corridor=is_in_corridor,
            predicted_corridor_entry=predicted_corridor_entry,
            object_class=track_obj.class_name,
            confidence=track_obj.confidence,
            stability_score=track_obj.stability_score,
            warning_triggered=warning_triggered
        )

    def assess_scene(
        self,
        active_tracks: List[TrackedObject],
        predictions: Dict[int, TrajectoryPrediction],
        frame_width: int,
        frame_height: int,
        current_timestamp: float
    ) -> SceneRiskSummary:
        """
        Aggregates assessments for all active tracks into a single scene-level risk summary.
        """
        if not active_tracks:
            return SceneRiskSummary(
                overall_risk_level=RiskLevel.NO_TARGETS_DETECTED,
                highest_risk_object_id=None,
                highest_risk_reason="No road users currently detected in field of view (uncalibrated camera view; scene is not guaranteed safe).",
                active_objects_count=0,
                assessments=[],
                timestamp=current_timestamp,
                scene_reliability="EMPTY_NO_TARGETS"
            )

        assessments: List[ObjectRiskAssessment] = []
        for track in active_tracks:
            pred = predictions.get(
                track.track_id,
                TrajectoryPrediction(
                    track_id=track.track_id,
                    velocity_xy=(0.0, 0.0),
                    speed_pixels_sec=0.0,
                    heading_deg=0.0,
                    motion_direction="Unknown",
                    predicted_points=[],
                    horizon_sec=1.5,
                    area_growth_rate=0.0,
                    is_reliable=False
                )
            )
            assessment = self.assess_object(track, pred, frame_width, frame_height, current_timestamp)
            assessments.append(assessment)

        # Sort assessments by risk score descending
        assessments.sort(key=lambda a: a.risk_score, reverse=True)
        top = assessments[0]

        return SceneRiskSummary(
            overall_risk_level=top.risk_level,
            highest_risk_object_id=top.track_id if top.risk_level != RiskLevel.SAFE else None,
            highest_risk_reason=top.reason,
            active_objects_count=len(active_tracks),
            assessments=assessments,
            timestamp=current_timestamp,
            scene_reliability="NORMAL"
        )
