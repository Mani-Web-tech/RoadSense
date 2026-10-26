"""
RoadSense Session Service
Tracks metrics throughout an analysis run, records warning event logs,
and provides structured JSON and CSV export capabilities.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import json
import io
import pandas as pd
import time

from utils.logging_config import logger


@dataclass
class WarningEventRecord:
    """Record of a single risk alert event triggered during analysis."""
    timestamp_sec: float
    formatted_time: str
    track_id: Optional[int]
    object_class: str
    risk_level: str
    risk_score: float
    reason: str
    uncalibrated_ttc_sec: Optional[float]


class SessionService:
    """
    Maintains session state, counts detections by class, logs warning events,
    and formats session summaries for display and export.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset all session counters for a new analysis run."""
        self.start_wall_time: float = time.time()
        self.end_wall_time: Optional[float] = None
        self.total_frames_processed: int = 0
        self.video_duration_sec: float = 0.0
        self.processed_duration_sec: float = 0.0
        self.unique_track_ids: set = set()
        self.class_detection_counts: Dict[str, int] = {}
        self.warning_events: List[WarningEventRecord] = []
        self.fps_measurements: List[float] = []
        self.latency_ms_measurements: List[float] = []
        self.system_errors: List[str] = []

    def record_frame(
        self,
        frame_timestamp: float,
        fps_instant: float,
        latency_ms: float,
        active_track_ids: List[int],
        active_classes: List[str]
    ) -> None:
        """Records frame-level processing metrics."""
        self.total_frames_processed += 1
        self.processed_duration_sec = frame_timestamp
        self.fps_measurements.append(fps_instant)
        self.latency_ms_measurements.append(latency_ms)

        for tid in active_track_ids:
            self.unique_track_ids.add(tid)

        for cls_name in active_classes:
            self.class_detection_counts[cls_name] = self.class_detection_counts.get(cls_name, 0) + 1

    def record_warning(
        self,
        timestamp_sec: float,
        track_id: Optional[int],
        object_class: str,
        risk_level: str,
        risk_score: float,
        reason: str,
        uncalibrated_ttc_sec: Optional[float]
    ) -> None:
        """Logs a warning event."""
        mins = int(timestamp_sec // 60)
        secs = timestamp_sec % 60
        formatted_time = f"{mins:02d}:{secs:05.2f}"

        record = WarningEventRecord(
            timestamp_sec=round(timestamp_sec, 2),
            formatted_time=formatted_time,
            track_id=track_id,
            object_class=object_class,
            risk_level=risk_level,
            risk_score=round(risk_score, 2),
            reason=reason,
            uncalibrated_ttc_sec=uncalibrated_ttc_sec
        )
        self.warning_events.append(record)

    def record_error(self, message: str) -> None:
        """Logs a runtime or processing error."""
        self.system_errors.append(message)
        logger.error(f"Session error logged: {message}")

    def finalize(self, total_video_duration: Optional[float] = None) -> None:
        """Finalizes session timing."""
        self.end_wall_time = time.time()
        if total_video_duration is not None and total_video_duration > 0:
            self.video_duration_sec = total_video_duration
        else:
            self.video_duration_sec = self.processed_duration_sec

    def get_summary_dict(self) -> Dict[str, Any]:
        """Returns consolidated summary statistics."""
        wall_duration = (
            (self.end_wall_time - self.start_wall_time)
            if self.end_wall_time else (time.time() - self.start_wall_time)
        )
        avg_fps = float(pd.Series(self.fps_measurements).mean()) if self.fps_measurements else 0.0
        avg_latency = float(pd.Series(self.latency_ms_measurements).mean()) if self.latency_ms_measurements else 0.0

        return {
            "session_status": "COMPLETED",
            "video_duration_sec": round(self.video_duration_sec, 2),
            "processed_duration_sec": round(self.processed_duration_sec, 2),
            "wall_clock_time_sec": round(wall_duration, 2),
            "total_frames_processed": self.total_frames_processed,
            "average_fps": round(avg_fps, 1),
            "average_latency_ms": round(avg_latency, 1),
            "unique_tracking_ids_count": len(self.unique_track_ids),
            "tracking_id_disclaimer": (
                "Tracking IDs reflect consecutive track associations, not guaranteed distinct real-world vehicles. "
                "ID reassignments or occlusions can create multiple track IDs for the same object."
            ),
            "detections_by_class": self.class_detection_counts,
            "warning_events_count": len(self.warning_events),
            "system_errors_count": len(self.system_errors),
            "system_errors": self.system_errors,
        }

    def export_summary_json(self) -> str:
        """Exports session summary and event logs as a JSON string."""
        data = self.get_summary_dict()
        data["warning_events"] = [asdict(ev) for ev in self.warning_events]
        return json.dumps(data, indent=2)

    def export_events_csv(self) -> str:
        """Exports warning events as CSV formatted text."""
        if not self.warning_events:
            df = pd.DataFrame(columns=[
                "timestamp_sec", "formatted_time", "track_id", "object_class",
                "risk_level", "risk_score", "reason", "uncalibrated_ttc_sec"
            ])
        else:
            df = pd.DataFrame([asdict(ev) for ev in self.warning_events])
        return df.to_csv(index=False)

    def export_summary_csv(self) -> str:
        """Exports key summary metrics as CSV."""
        summary = self.get_summary_dict()
        rows = []
        for k, v in summary.items():
            if k not in ("detections_by_class", "system_errors"):
                rows.append({"Metric": k, "Value": str(v)})
        for cls_name, count in self.class_detection_counts.items():
            rows.append({"Metric": f"Detections ({cls_name})", "Value": str(count)})
        return pd.DataFrame(rows).to_csv(index=False)
