"""
RoadSense Validation Module
Validates inputs, configuration parameters, and media files with informative error handling.
"""
import os
from typing import Tuple, List
from config import AppConfig

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_VIDEO_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB limit for prototype safety


class RoadSenseValidationError(ValueError):
    """Custom exception raised when an input or configuration parameter is invalid."""
    pass


def validate_video_file(file_path: str) -> Tuple[bool, str]:
    """
    Validates that a video file exists, is non-empty, has an allowed extension,
    and does not exceed safety size limits.

    Returns:
        (is_valid, error_or_success_message)
    """
    if not file_path:
        return False, "Video file path is empty."

    if not os.path.exists(file_path):
        return False, f"File not found at path: {file_path}"

    if not os.path.isfile(file_path):
        return False, f"Specified path is not a regular file: {file_path}"

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        return False, f"Unsupported video format '{ext}'. Supported formats: {allowed}"

    try:
        size = os.path.getsize(file_path)
        if size == 0:
            return False, "Video file is empty (0 bytes)."
        if size > MAX_VIDEO_FILE_SIZE_BYTES:
            max_mb = MAX_VIDEO_FILE_SIZE_BYTES // (1024 * 1024)
            return False, f"File exceeds maximum allowed size ({max_mb} MB)."
    except OSError as e:
        return False, f"Cannot access file metadata: {e}"

    return True, "Video file is valid."


def validate_config(config: AppConfig) -> List[str]:
    """
    Validates application configuration parameters.
    Returns a list of validation error descriptions (empty if valid).
    """
    errors: List[str] = []

    # Detector checks
    if not (0.05 <= config.detector.confidence_threshold <= 1.0):
        errors.append(f"Confidence threshold must be between 0.05 and 1.0 (got {config.detector.confidence_threshold}).")

    if not (0.05 <= config.detector.iou_threshold <= 1.0):
        errors.append(f"IoU threshold must be between 0.05 and 1.0 (got {config.detector.iou_threshold}).")

    # Trajectory checks
    if not (0.2 <= config.trajectory.prediction_horizon_sec <= 10.0):
        errors.append(f"Prediction horizon must be between 0.2s and 10.0s (got {config.trajectory.prediction_horizon_sec}s).")

    if config.trajectory.min_observations < 2:
        errors.append(f"Minimum observations must be at least 2 (got {config.trajectory.min_observations}).")

    # Risk checks
    if not (0.0 <= config.risk.ego_corridor_left < config.risk.ego_corridor_right <= 1.0):
        errors.append(
            f"Ego corridor bounds invalid: left ({config.risk.ego_corridor_left}) must be < right ({config.risk.ego_corridor_right}) within [0, 1]."
        )

    if config.risk.warning_cooldown_sec < 0.0:
        errors.append(f"Warning cooldown cannot be negative (got {config.risk.warning_cooldown_sec}).")

    if config.risk.heuristic_ttc_high_sec <= 0:
        errors.append("High risk TTC threshold must be greater than 0.")

    return errors
