"""
Tests for Validation Module
Verifies video file checks, size limits, format restrictions, and configuration boundaries.
"""
import os
import tempfile
import pytest
from utils.validation import validate_video_file, validate_config
from config import AppConfig


def test_validate_video_file_empty_path():
    valid, msg = validate_video_file("")
    assert not valid
    assert "empty" in msg.lower()


def test_validate_video_file_nonexistent():
    valid, msg = validate_video_file("nonexistent_video_path_xyz123.mp4")
    assert not valid
    assert "not found" in msg.lower()


def test_validate_video_file_invalid_extension():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"not a video")
        temp_path = f.name

    try:
        valid, msg = validate_video_file(temp_path)
        assert not valid
        assert "unsupported video format" in msg.lower()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_validate_video_file_empty_file():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        temp_path = f.name

    try:
        valid, msg = validate_video_file(temp_path)
        assert not valid
        assert "empty (0 bytes)" in msg.lower()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_validate_video_file_valid_mp4():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"\x00\x00\x00\x20ftypmp42" + b"dummy data" * 100)
        temp_path = f.name

    try:
        valid, msg = validate_video_file(temp_path)
        assert valid
        assert "valid" in msg.lower()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_validate_config_valid():
    cfg = AppConfig()
    errors = validate_config(cfg)
    assert len(errors) == 0


def test_validate_config_invalid_thresholds():
    cfg = AppConfig()
    cfg.detector.confidence_threshold = 1.5  # Invalid (> 1.0)
    cfg.detector.iou_threshold = -0.1        # Invalid (< 0.05)
    cfg.trajectory.prediction_horizon_sec = 0.0  # Invalid
    cfg.risk.ego_corridor_left = 0.8
    cfg.risk.ego_corridor_right = 0.2        # Invalid (left > right)

    errors = validate_config(cfg)
    assert len(errors) >= 4
