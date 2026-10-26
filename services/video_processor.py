"""
RoadSense Video Processing Pipeline
Integrates tracking, trajectory prediction, collision-risk assessment, and OpenCV overlays.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time
import math
import cv2
import numpy as np

from config import AppConfig
from camera.base_source import FrameData
from models.tracker import RoadUserTracker, TrackedObject
from models.trajectory_predictor import TrajectoryPredictor, TrajectoryPrediction
from models.risk_assessor import RiskAssessor, SceneRiskSummary, ObjectRiskAssessment, RiskLevel
from utils.logging_config import logger


# BGR Color definitions for OpenCV annotations
COLOR_HIGH_RISK = (40, 40, 235)       # Crimson Red
COLOR_MEDIUM_RISK = (0, 140, 255)     # Vibrant Amber / Orange
COLOR_LOW_RISK = (0, 215, 255)        # Warning Yellow
COLOR_SAFE = (70, 220, 80)            # Safety Green
COLOR_INSUFFICIENT = (150, 150, 150)  # Neutral Gray
COLOR_CORRIDOR = (200, 200, 200)      # Corridor boundary


@dataclass
class ProcessedFrameResult:
    """Consolidated outcome of processing a single video frame."""
    frame_index: int
    timestamp: float
    annotated_frame: np.ndarray
    scene_summary: SceneRiskSummary
    active_tracks: List[TrackedObject]
    predictions: Dict[int, TrajectoryPrediction]
    assessments: List[ObjectRiskAssessment]
    latency_ms: float
    fps_instant: float
    triggered_warnings: List[ObjectRiskAssessment]


class VideoProcessor:
    """
    Core processing engine coordinating tracking, trajectory forecasting,
    risk evaluation, and multi-layer HUD overlays.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.tracker = RoadUserTracker(self.config.detector, self.config.tracker)
        self.trajectory_predictor = TrajectoryPredictor(self.config.trajectory)
        self.risk_assessor = RiskAssessor(self.config.risk)
        self._last_process_time: Optional[float] = None

    def reset(self) -> None:
        """Reset internal pipeline state."""
        self.tracker.reset()
        self.risk_assessor.reset()
        self._last_process_time = None

    def process_frame(self, frame_data: FrameData) -> ProcessedFrameResult:
        """
        Executes full end-to-end processing pipeline on a single frame.
        """
        t_start = time.perf_counter()
        raw_frame = frame_data.frame
        h, w = raw_frame.shape[:2]
        timestamp = frame_data.timestamp

        # 1. Multi-object tracking
        active_tracks = self.tracker.track(raw_frame, timestamp)

        # 2. Trajectory prediction for each active track
        predictions: Dict[int, TrajectoryPrediction] = {}
        for track in active_tracks:
            pred = self.trajectory_predictor.predict(track.track_id, track.history, timestamp)
            predictions[track.track_id] = pred

        # 3. Collision risk evaluation
        scene_summary = self.risk_assessor.assess_scene(
            active_tracks, predictions, w, h, timestamp
        )

        triggered_warnings: List[ObjectRiskAssessment] = [
            a for a in scene_summary.assessments if a.warning_triggered
        ]

        # 4. Rendering annotations onto frame copy
        annotated = raw_frame.copy()

        # Draw ego corridor boundary if enabled
        if self.config.display_corridor:
            self._draw_ego_corridor(annotated, w, h)

        # Draw each tracked road user, trajectory trail, and predicted path
        assessments_by_id = {a.track_id: a for a in scene_summary.assessments}
        for track in active_tracks:
            assessment = assessments_by_id.get(track.track_id)
            pred = predictions.get(track.track_id)
            self._draw_object_overlay(annotated, track, assessment, pred)

        # Draw top HUD banner
        t_end = time.perf_counter()
        latency_ms = max(1.0, (t_end - t_start) * 1000.0)

        # Calculate processing FPS
        fps_instant = 1000.0 / latency_ms
        if self._last_process_time is not None:
            dt = t_end - self._last_process_time
            if dt > 0.001:
                fps_instant = 1.0 / dt
        self._last_process_time = t_end

        self._draw_hud(annotated, scene_summary, fps_instant, latency_ms, w, h)

        return ProcessedFrameResult(
            frame_index=frame_data.frame_index,
            timestamp=timestamp,
            annotated_frame=annotated,
            scene_summary=scene_summary,
            active_tracks=active_tracks,
            predictions=predictions,
            assessments=scene_summary.assessments,
            latency_ms=latency_ms,
            fps_instant=fps_instant,
            triggered_warnings=triggered_warnings
        )

    def _draw_ego_corridor(self, img: np.ndarray, w: int, h: int) -> None:
        """Draws subtle lane corridor guidelines indicating camera vehicle path."""
        c_left = int(w * self.config.risk.ego_corridor_left)
        c_right = int(w * self.config.risk.ego_corridor_right)
        horizon_y = int(h * 0.45)

        # Vanishing perspective corridor
        vanishing_x = int(w * 0.5)
        pts_left = np.array([[vanishing_x - 30, horizon_y], [c_left, h]], np.int32)
        pts_right = np.array([[vanishing_x + 30, horizon_y], [c_right, h]], np.int32)

        # Overlay transparent corridor floor
        corridor_poly = np.array([
            [vanishing_x - 30, horizon_y],
            [vanishing_x + 30, horizon_y],
            [c_right, h],
            [c_left, h]
        ], np.int32)

        overlay = img.copy()
        cv2.fillPoly(overlay, [corridor_poly], (40, 70, 40))
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

        # Draw guideline borders
        cv2.polylines(img, [pts_left], False, COLOR_CORRIDOR, 1, cv2.LINE_AA)
        cv2.polylines(img, [pts_right], False, COLOR_CORRIDOR, 1, cv2.LINE_AA)

    def _draw_object_overlay(
        self,
        img: np.ndarray,
        track: TrackedObject,
        assessment: Optional[ObjectRiskAssessment],
        pred: Optional[TrajectoryPrediction]
    ) -> None:
        """Draws bounding box, label badge, past trajectory trail, and future predicted path."""
        # Determine color by risk level
        risk_lvl = assessment.risk_level if assessment else RiskLevel.SAFE
        if risk_lvl == RiskLevel.HIGH:
            color = COLOR_HIGH_RISK
            thickness = 3
        elif risk_lvl == RiskLevel.MEDIUM:
            color = COLOR_MEDIUM_RISK
            thickness = 2
        elif risk_lvl == RiskLevel.LOW:
            color = COLOR_LOW_RISK
            thickness = 2
        elif risk_lvl == RiskLevel.INSUFFICIENT_DATA:
            color = COLOR_INSUFFICIENT
            thickness = 1
        else:
            color = COLOR_SAFE
            thickness = 2

        x1, y1, x2, y2 = map(int, track.box)
        w_box = x2 - x1
        h_box = y2 - y1

        # Draw bounding box corners / rect
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Draw past trajectory trail (solid dots & line)
        if self.config.display_trails and len(track.history) > 1:
            pts = [(int(pt[0]), int(pt[1])) for pt in track.history]
            for i in range(1, len(pts)):
                cv2.line(img, pts[i - 1], pts[i], (200, 230, 200), 2, cv2.LINE_AA)
            for pt in pts[-5:]:
                cv2.circle(img, pt, 3, (120, 240, 120), -1)

        # Draw future predicted trajectory (magenta/yellow dashed path with target markers)
        if self.config.display_predictions and pred and pred.is_reliable and pred.predicted_points:
            pred_pts = [(int(p[0]), int(p[1])) for p in pred.predicted_points]
            curr_pt = (int(track.center[0]), int(track.center[1]))
            all_forecast = [curr_pt] + pred_pts

            for i in range(1, len(all_forecast)):
                # Draw dashed / segmented projection line
                cv2.line(img, all_forecast[i - 1], all_forecast[i], (255, 105, 180), 2, cv2.LINE_AA)
                cv2.circle(img, all_forecast[i], 3, (255, 180, 220), -1)

            # Draw trajectory horizon arrowhead
            if len(pred_pts) > 0:
                cv2.drawMarker(img, pred_pts[-1], (255, 50, 180), markerType=cv2.MARKER_TRIANGLE_UP, markerSize=8, thickness=2)

        # Draw label badge above bounding box
        label_text = f"#{track.track_id} {track.class_name.upper()} {int(track.confidence * 100)}%"
        if assessment and assessment.risk_level in (RiskLevel.HIGH, RiskLevel.MEDIUM):
            label_text += f" [{assessment.risk_level}]"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        (lw, lh), baseline = cv2.getTextSize(label_text, font, font_scale, 1)

        badge_y = max(y1 - 5, lh + 6)
        cv2.rectangle(img, (x1, badge_y - lh - 4), (x1 + lw + 6, badge_y + 2), color, -1)
        cv2.putText(img, label_text, (x1 + 3, badge_y - 2), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_hud(
        self,
        img: np.ndarray,
        summary: SceneRiskSummary,
        fps: float,
        latency_ms: float,
        w: int,
        h: int
    ) -> None:
        """Draws top HUD banner with risk status, object count, and processing telemetry."""
        hud_h = 42
        # Semi-transparent dark banner overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

        # Risk indicator pill color
        if summary.overall_risk_level == RiskLevel.HIGH:
            badge_color = COLOR_HIGH_RISK
            badge_text = "HIGH RISK WARNING"
        elif summary.overall_risk_level == RiskLevel.MEDIUM:
            badge_color = COLOR_MEDIUM_RISK
            badge_text = "CAUTION: MEDIUM RISK"
        elif summary.overall_risk_level == RiskLevel.LOW:
            badge_color = COLOR_LOW_RISK
            badge_text = "LOW RISK - MONITORING"
        elif summary.overall_risk_level == RiskLevel.NO_TARGETS_DETECTED:
            badge_color = COLOR_INSUFFICIENT
            badge_text = "NO TARGETS DETECTED"
        elif summary.overall_risk_level == RiskLevel.INSUFFICIENT_DATA:
            badge_color = COLOR_INSUFFICIENT
            badge_text = "INSUFFICIENT DATA"
        else:
            badge_color = COLOR_SAFE
            badge_text = "STATUS: SAFE / RECEDING"

        # Draw status pill
        cv2.rectangle(img, (12, 8), (240, 34), badge_color, -1)
        cv2.putText(img, badge_text, (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Telemetry metrics on right
        telemetry_text = f"Users: {summary.active_objects_count}  |  FPS: {fps:.1f}  |  Latency: {latency_ms:.0f}ms"
        (tw, th), _ = cv2.getTextSize(telemetry_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(img, telemetry_text, (w - tw - 15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
