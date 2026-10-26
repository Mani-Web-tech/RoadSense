"""
RoadSense Video File Source
Implements frame ingestion from recorded video files (MP4, AVI, MOV, etc.).
"""
import os
import time
from typing import Optional, Dict, Any
import cv2
import numpy as np

from camera.base_source import BaseCameraSource, FrameData
from utils.logging_config import logger
from utils.validation import validate_video_file


class VideoFileSource(BaseCameraSource):
    """
    Video capture from local media files.
    Reads frame-by-frame, avoiding loading entire videos into memory.
    """

    def __init__(self, file_path: str, target_width: Optional[int] = None):
        self.file_path = file_path
        self.target_width = target_width
        self.cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._fps = 30.0
        self._total_frames = 0
        self._width = 0
        self._height = 0
        self._is_opened = False

    def open(self) -> bool:
        is_valid, msg = validate_video_file(self.file_path)
        if not is_valid:
            logger.error(f"Cannot open video source: {msg}")
            return False

        try:
            self.cap = cv2.VideoCapture(self.file_path)
            if not self.cap.isOpened():
                logger.error(f"cv2.VideoCapture failed to open file: {self.file_path}")
                self._is_opened = False
                return False

            self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            if self._fps <= 0 or np.isnan(self._fps):
                self._fps = 30.0

            self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            self._frame_count = 0
            self._is_opened = True

            logger.info(
                f"Opened video file '{os.path.basename(self.file_path)}': "
                f"{self._width}x{self._height} @ {self._fps:.1f} FPS, {self._total_frames} frames."
            )
            return True

        except Exception as e:
            logger.error(f"Unexpected error opening video file: {e}", exc_info=True)
            self._is_opened = False
            return False

    def read_frame(self) -> Optional[FrameData]:
        if not self._is_opened or self.cap is None:
            return None

        success, frame = self.cap.read()
        if not success or frame is None:
            logger.info("End of video stream reached.")
            return None

        self._frame_count += 1
        timestamp = self._frame_count / self._fps

        # Optional aspect-ratio-preserving resize
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
                "fps": self._fps,
                "total_frames": self._total_frames,
                "original_width": self._width,
                "original_height": self._height,
                "current_width": frame.shape[1],
                "current_height": frame.shape[0],
            }
        )

    def release(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
                logger.info(f"Released video file: {self.file_path}")
            except Exception as e:
                logger.warning(f"Error during video release: {e}")
            finally:
                self.cap = None
        self._is_opened = False

    def is_opened(self) -> bool:
        return self._is_opened and self.cap is not None and self.cap.isOpened()

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source_type": "file",
            "file_path": self.file_path,
            "fps": self._fps,
            "total_frames": self._total_frames,
            "duration_sec": (self._total_frames / self._fps) if self._fps > 0 else 0.0,
            "width": self._width,
            "height": self._height,
        }
