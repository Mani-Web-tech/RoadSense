"""
RoadSense Object Detection Module
Wraps Ultralytics YOLO to detect targeted road-user classes:
cars, motorcycles, bicycles, buses, trucks, and pedestrians.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
from ultralytics import YOLO

from config import DetectorConfig, COCO_ROAD_USERS
from utils.logging_config import logger


@dataclass
class Detection:
    """Standardized road-user detection object."""
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    center: Tuple[float, float]             # (cx, cy)
    area: float                             # width * height in pixels


class RoadUserDetector:
    """
    Performs road-user detection using Ultralytics YOLO models.
    Supports graceful fallback and model availability validation.
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.model: Optional[YOLO] = None
        self._is_ready = False
        self._load_model()

    def _load_model(self) -> None:
        try:
            logger.info(f"Loading object detection model: {self.config.model_name}")
            self.model = YOLO(self.config.model_name)
            self._is_ready = True
            logger.info(f"Successfully loaded detector: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model '{self.config.model_name}': {e}", exc_info=True)
            self.model = None
            self._is_ready = False

    def is_available(self) -> bool:
        """Returns True if the underlying model is successfully loaded and ready for inference."""
        return self._is_ready and self.model is not None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Runs object detection on a single frame.

        Args:
            frame: BGR numpy image.

        Returns:
            List of validated Detection objects matching target road-user classes.
        """
        if not self.is_available() or frame is None:
            return []

        detections: List[Detection] = []
        try:
            results = self.model(
                frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                classes=self.config.target_classes,
                device=self.config.device,
                verbose=False
            )

            if not results or len(results) == 0:
                return []

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return []

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(xyxy)):
                cls_id = int(classes[i])
                conf = float(confs[i])
                x1, y1, x2, y2 = map(float, xyxy[i])
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                area = w * h

                # Filter out microscopic noise boxes
                if area < 100.0:
                    continue

                cx = x1 + (w / 2.0)
                cy = y1 + (h / 2.0)
                cls_name = COCO_ROAD_USERS.get(cls_id, f"class_{cls_id}")

                detections.append(Detection(
                    box=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    center=(cx, cy),
                    area=area
                ))

        except Exception as e:
            logger.error(f"Inference error during detection: {e}", exc_info=True)

        return detections
