"""
RoadSense Object Tracking Module
Tracks detected road users across consecutive frames using ByteTrack.
Maintains identity, center position histories with timestamps, and track stability.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
from ultralytics import YOLO

from config import TrackerConfig, DetectorConfig, COCO_ROAD_USERS
from utils.logging_config import logger


@dataclass
class TrackedObject:
    """Represents a consistently tracked road user across consecutive frames."""
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: Tuple[float, float]             # (cx, cy)
    area: float
    # History stores list of (cx, cy, timestamp, area)
    history: List[Tuple[float, float, float, float]] = field(default_factory=list)
    first_seen_timestamp: float = 0.0
    last_seen_timestamp: float = 0.0
    lost_frames: int = 0
    consecutive_hits: int = 1

    @property
    def stability_score(self) -> float:
        """
        Estimates track stability based on observations and consecutive hits.
        Normalized between 0.0 (unstable/new) and 1.0 (highly reliable).
        """
        obs_count = len(self.history)
        if obs_count < 3:
            return 0.25
        elif obs_count < 6:
            return 0.60
        elif obs_count < 12:
            return 0.85
        return 1.0


class RoadUserTracker:
    """
    Manages multi-object tracking across video frames.
    Uses ByteTrack through Ultralytics tracking engine.
    """

    def __init__(self, detector_config: Optional[DetectorConfig] = None, tracker_config: Optional[TrackerConfig] = None):
        self.det_config = detector_config or DetectorConfig()
        self.track_config = tracker_config or TrackerConfig()
        self.model: Optional[YOLO] = None
        self.active_tracks: Dict[int, TrackedObject] = {}
        self._total_unique_tracks_observed: set = set()
        self._is_ready = False
        self._load_model()

    def _load_model(self) -> None:
        try:
            logger.info("Initializing tracking model with ByteTrack...")
            self.model = YOLO(self.det_config.model_name)
            self._is_ready = True
            logger.info("Tracker initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize tracker: {e}", exc_info=True)
            self._is_ready = False

    def is_available(self) -> bool:
        return self._is_ready and self.model is not None

    def reset(self) -> None:
        """Resets tracking state (e.g., when starting a new video stream)."""
        self.active_tracks.clear()
        self._total_unique_tracks_observed.clear()
        if self.model is not None:
            # Clear tracker cache if supported
            try:
                self.model.predictor = None
            except Exception:
                pass
        logger.info("Tracker state reset.")

    def track(self, frame: np.ndarray, timestamp: float) -> List[TrackedObject]:
        """
        Tracks road users in the provided frame.

        Args:
            frame: BGR numpy image.
            timestamp: Frame timestamp in seconds.

        Returns:
            List of currently active TrackedObject instances.
        """
        if not self.is_available() or frame is None:
            return []

        current_frame_tracks: List[TrackedObject] = []
        observed_ids: set = set()

        try:
            results = self.model.track(
                frame,
                persist=True,
                conf=self.det_config.confidence_threshold,
                iou=self.det_config.iou_threshold,
                classes=self.det_config.target_classes,
                device=self.det_config.device,
                tracker=self.track_config.tracker_type,
                verbose=False
            )

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)

                # Check if tracking IDs were assigned
                track_ids = None
                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy().astype(int)

                for i in range(len(xyxy)):
                    # If tracker didn't assign an ID (e.g. single frame unassociated), assign fallback
                    t_id = int(track_ids[i]) if (track_ids is not None and i < len(track_ids)) else (99000 + i)
                    observed_ids.add(t_id)
                    self._total_unique_tracks_observed.add(t_id)

                    cls_id = int(classes[i])
                    conf = float(confs[i])
                    x1, y1, x2, y2 = map(float, xyxy[i])
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    area = w * h

                    if area < self.track_config.min_box_area:
                        continue

                    cx = x1 + (w / 2.0)
                    cy = y1 + (h / 2.0)
                    cls_name = COCO_ROAD_USERS.get(cls_id, f"class_{cls_id}")

                    if t_id in self.active_tracks:
                        track_obj = self.active_tracks[t_id]
                        track_obj.box = (x1, y1, x2, y2)
                        track_obj.center = (cx, cy)
                        track_obj.area = area
                        track_obj.confidence = conf
                        track_obj.last_seen_timestamp = timestamp
                        track_obj.lost_frames = 0
                        track_obj.consecutive_hits += 1
                        track_obj.history.append((cx, cy, timestamp, area))
                        # Limit history size
                        if len(track_obj.history) > self.track_config.track_history_len:
                            track_obj.history = track_obj.history[-self.track_config.track_history_len:]
                    else:
                        track_obj = TrackedObject(
                            track_id=t_id,
                            class_id=cls_id,
                            class_name=cls_name,
                            confidence=conf,
                            box=(x1, y1, x2, y2),
                            center=(cx, cy),
                            area=area,
                            history=[(cx, cy, timestamp, area)],
                            first_seen_timestamp=timestamp,
                            last_seen_timestamp=timestamp,
                            lost_frames=0,
                            consecutive_hits=1
                        )
                        self.active_tracks[t_id] = track_obj

                    current_frame_tracks.append(track_obj)

            # Update lost frames for tracks not observed in this frame
            to_remove = []
            for t_id, track_obj in self.active_tracks.items():
                if t_id not in observed_ids:
                    track_obj.lost_frames += 1
                    track_obj.consecutive_hits = 0
                    if track_obj.lost_frames > 15:  # Prune after 15 lost frames
                        to_remove.append(t_id)

            for t_id in to_remove:
                del self.active_tracks[t_id]

        except Exception as e:
            logger.error(f"Error during tracking execution: {e}", exc_info=True)

        return current_frame_tracks

    def get_unique_tracks_count(self) -> int:
        """Returns the total number of unique track IDs seen in the current session."""
        return len(self._total_unique_tracks_observed)
