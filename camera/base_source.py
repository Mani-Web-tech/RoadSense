"""
RoadSense Camera Sources Package
Provides unified frame ingestion from recorded video, webcam, simulated streams,
and vehicle camera SDK interfaces.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class FrameData:
    """Standardized representation of a single captured frame and its acquisition telemetry."""
    frame: np.ndarray
    timestamp: float           # Relative or absolute epoch time in seconds
    frame_index: int           # Monotonically increasing frame sequence number
    source_status: str         # "ACTIVE", "EOS", "ERROR", "PAUSED"
    metadata: Dict[str, Any] = field(default_factory=dict)  # FPS, camera intrinsics if available, etc.


class BaseCameraSource(ABC):
    """
    Abstract interface for all video/camera sources.
    Decouples computer vision processing from hardware-specific acquisition details.
    """

    @abstractmethod
    def open(self) -> bool:
        """Initialize and open the video/camera stream."""
        pass

    @abstractmethod
    def read_frame(self) -> Optional[FrameData]:
        """
        Fetch the next frame from the stream.
        Returns FrameData if available, or None when end of stream / error is reached.
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Safely release underlying hardware or operating system file handles."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Check if source is currently connected and active."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return stream metadata such as resolution, FPS, total frame count."""
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
