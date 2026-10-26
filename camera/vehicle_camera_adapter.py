"""
RoadSense Vehicle Camera Integration Adapter
Architectural boundary and interface for future manufacturer-approved vehicle camera SDKs.

========================================================================================
CRITICAL AUTOMOTIVE INTEGRATION BOUNDARY NOTICE:
Commercial factory-installed vehicle cameras (ADAS / surround-view / backup) are gated
behind strict OEM hardware security modules, ASIL safety boundaries, and proprietary HALs.
Standard userland applications (Android Auto projection, standard Android Automotive OS,
or third-party desktop apps) DO NOT have raw, direct access to these feeds without:
 1. Manufacturer-approved cryptographic signing and OEM developer agreements.
 2. Hardware Abstraction Layer (HAL) / Car Service bindings (e.g., EVS - Exterior View System).
 3. ASIL-B / ASIL-D functional safety clearance.
 4. Calibrated camera intrinsics/extrinsics and vehicle CAN-bus telemetry.

This adapter establishes the formal contract that an authorized OEM SDK must satisfy.
========================================================================================
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from camera.base_source import BaseCameraSource, FrameData
from utils.logging_config import logger


@dataclass
class VehicleIntegrationSpec:
    """Documented specification for future vehicle-specific hardware integration."""
    target_make: str = "Generic OEM"
    target_model: str = "Demonstrator"
    model_year: int = 2026
    operating_system: str = "Android Automotive OS 14+ (EVS Service) / QNX Neutrino"
    is_projection_mode: bool = False  # True if Android Auto / CarPlay (which strictly forbid camera access)
    manufacturer_api_name: str = "OEM CarCameraService / EVS HAL"
    required_permissions: str = "android.car.permission.CAR_CAMERA_SERVICE / OEM_ADAS_READ"
    camera_stream_id: str = "front_adas_narrow_fov"
    resolution: tuple = (1920, 1080)
    fps_capability: float = 30.0
    telemetry_channel_available: bool = False  # Vehicle speed, steering angle, yaw rate


class VehicleCameraAdapter(BaseCameraSource):
    """
    Integration adapter for manufacturer-authorized vehicle camera systems.

    In production: Bridges to OEM-certified native libraries (JNI/C++ EVS client).
    In this prototype: Provides a clearly-bounded integration harness with simulation
    mode for pipeline validation, while raising explicit status notices that raw OEM
    hardware is unattached.
    """

    def __init__(self, spec: Optional[VehicleIntegrationSpec] = None, simulation_mode: bool = False):
        self.spec = spec or VehicleIntegrationSpec()
        self.simulation_mode = simulation_mode
        self._is_opened = False
        self._frame_count = 0
        self._last_timestamp = 0.0

    def open(self) -> bool:
        logger.info(
            f"Initializing VehicleCameraAdapter: Make={self.spec.target_make}, "
            f"OS={self.spec.operating_system}, Stream={self.spec.camera_stream_id}"
        )

        if not self.simulation_mode:
            logger.warning(
                "VEHICLE ADAPTER NOTICE: No authorized OEM hardware bridge is present on this host. "
                "Factory vehicle cameras cannot be accessed on generic consumer workstations without "
                "manufacturer-approved SDKs and hardware loop connectors."
            )
            # Fails gracefully rather than pretending to connect
            self._is_opened = False
            return False

        # Simulation mode allowed for integration pipeline testing
        logger.info("VehicleCameraAdapter running in simulated bench-test mode.")
        self._is_opened = True
        self._frame_count = 0
        return True

    def read_frame(self) -> Optional[FrameData]:
        if not self._is_opened:
            return None

        self._frame_count += 1
        timestamp = self._frame_count / self.spec.fps_capability

        # Generate a synthetic benchmark test frame for verification
        width, height = self.spec.resolution
        frame = np.zeros((min(height, 480), min(width, 640), 3), dtype=np.uint8)

        # Draw simulated vehicle test pattern
        frame[:] = (30, 30, 35)

        return FrameData(
            frame=frame,
            timestamp=timestamp,
            frame_index=self._frame_count,
            source_status="SIMULATED_VEHICLE_STREAM",
            metadata={
                "oem_stream_id": self.spec.camera_stream_id,
                "fps": self.spec.fps_capability,
                "os": self.spec.operating_system,
                "telemetry_present": self.spec.telemetry_channel_available,
            }
        )

    def release(self) -> None:
        self._is_opened = False
        logger.info("Vehicle camera adapter session closed.")

    def is_opened(self) -> bool:
        return self._is_opened

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source_type": "vehicle_camera_adapter",
            "make": self.spec.target_make,
            "os": self.spec.operating_system,
            "stream_id": self.spec.camera_stream_id,
            "simulation_mode": self.simulation_mode,
            "boundary_status": "OEM_AUTHENTICATION_REQUIRED",
        }
