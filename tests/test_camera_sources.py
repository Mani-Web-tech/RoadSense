"""
Tests for Camera Source Implementations
Verifies video files, webcams, failure handling, and vehicle camera adapters.
"""
import pytest
from camera.video_source import VideoFileSource
from camera.webcam_source import WebcamSource
from camera.vehicle_camera_adapter import VehicleCameraAdapter, VehicleIntegrationSpec
from camera.base_source import FrameData


def test_video_file_source_invalid_path():
    src = VideoFileSource("nonexistent_video_path.mp4")
    assert not src.open()
    assert src.read_frame() is None
    assert not src.is_opened()
    src.release()


def test_video_file_source_valid_sample():
    src = VideoFileSource("sample_road.mp4", target_width=480)
    assert src.open()
    assert src.is_opened()

    frame_data = src.read_frame()
    assert frame_data is not None
    assert isinstance(frame_data, FrameData)
    assert frame_data.frame is not None
    assert frame_data.frame.shape[1] == 480  # Resized width
    assert frame_data.source_status == "ACTIVE"

    meta = src.get_metadata()
    assert meta["source_type"] == "file"
    assert meta["fps"] > 0

    src.release()
    assert not src.is_opened()


def test_webcam_source_invalid_device():
    # Device index 999 is guaranteed unavailable on typical hosts
    cam = WebcamSource(device_index=999)
    assert not cam.open()
    assert cam.read_frame() is None
    cam.release()


def test_vehicle_camera_adapter_hardware_boundary():
    # Without simulation_mode, factory camera must not claim connection
    adapter = VehicleCameraAdapter(simulation_mode=False)
    assert not adapter.open()
    assert not adapter.is_opened()
    assert adapter.read_frame() is None

    meta = adapter.get_metadata()
    assert meta["boundary_status"] == "OEM_AUTHENTICATION_REQUIRED"


def test_vehicle_camera_adapter_simulation_mode():
    adapter = VehicleCameraAdapter(simulation_mode=True)
    assert adapter.open()
    assert adapter.is_opened()

    frame_data = adapter.read_frame()
    assert frame_data is not None
    assert frame_data.source_status == "SIMULATED_VEHICLE_STREAM"
    assert frame_data.frame.shape[0] > 0

    adapter.release()
    assert not adapter.is_opened()
