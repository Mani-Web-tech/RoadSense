"""
RoadSense Webcam / USB Camera Source
Provides real-time frame acquisition from connected cameras with graceful fallback.
"""
import time
from typing import Optional, Dict, Any
import cv2
import numpy as np

from camera.base_source import BaseCameraSource, FrameData
from utils.logging_config import logger


class WebcamSource(BaseCameraSource):
    """
    Live video capture from a local USB camera or integrated webcam.
    Gracefully detects unavailable devices and permission issues.
    """

    def __init__(self, device_index: int = 0, target_width: Optional[int] = None):
        self.device_index = device_index
        self.target_width = target_width
        self.cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._start_time: Optional[float] = None
        self._is_opened = False
        self._fps_estimate = 30.0

    def open(self) -> bool:
        try:
            logger.info(f"Attempting to open webcam device index: {self.device_index}")
            # On Windows, try cv2.CAP_DSHOW for faster initialization and clean capture
            self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default backend
                self.cap = cv2.VideoCapture(self.device_index)

            if not self.cap.isOpened():
                logger.warning(
                    f"Camera device {self.device_index} is unavailable, in use by another app, "
                    "or permission was denied."
                )
                self._is_opened = False
                return False

            # Test reading a single frame to ensure hardware readiness
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                logger.warning(f"Webcam {self.device_index} opened but returned no frame.")
                self.release()
                return False

            self._frame_count = 0
            self._start_time = time.time()
            self._is_opened = True
            logger.info(f"Successfully initialized live webcam {self.device_index}.")
            return True

        except Exception as e:
            logger.error(f"Error accessing webcam: {e}", exc_info=True)
            self._is_opened = False
            return False

    def read_frame(self) -> Optional[FrameData]:
        if not self._is_opened or self.cap is None:
            return None

        success, frame = self.cap.read()
        if not success or frame is None:
            logger.warning("Failed to grab frame from webcam stream.")
            return None

        self._frame_count += 1
        now = time.time()
        timestamp = (now - self._start_time) if self._start_time else (self._frame_count / 30.0)

        if self.target_width and self.target_width > 0 and frame.shape[1] > self.target_width:
            aspect = frame.shape[0] / frame.shape[1]
            new_h = int(self.target_width * aspect)
            frame = cv2.resize(frame, (self.target_width, new_h), interpolation=cv2.INTER_AREA)

        return FrameData(
            frame=frame,
            timestamp=timestamp,
            frame_index=self._frame_count,
            source_status="ACTIVE",
            metadata={
                "device_index": self.device_index,
                "current_width": frame.shape[1],
                "current_height": frame.shape[0],
            }
        )

    def release(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
                logger.info(f"Released webcam {self.device_index}.")
            except Exception as e:
                logger.warning(f"Error closing webcam: {e}")
            finally:
                self.cap = None
        self._is_opened = False

    def is_opened(self) -> bool:
        return self._is_opened and self.cap is not None and self.cap.isOpened()

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source_type": "webcam",
            "device_index": self.device_index,
            "status": "connected" if self.is_opened() else "disconnected",
        }
