# RoadSense — AI-Powered Road Safety & Collision Warning Application

> **"See the Risk. Predict the Danger. Protect the Journey."**

RoadSense is a computer-vision and machine-learning road safety application designed to detect road users, track their motion, forecast short-term future image-space trajectories, estimate collision conflict risk, and deliver multi-sensory visual and audio warnings before dangerous situations develop.

Developed as a desktop and browser-based engineering prototype, RoadSense incorporates a clean **Camera Source Abstraction Layer** allowing seamless transition from recorded road video and USB webcams to manufacturer-authorized vehicle camera SDKs and Android Automotive OS (AAOS) infotainment platforms.

---

## Key Features

- **Multi-Class Road User Detection:** Detects and classifies cars, motorcycles, bicycles, buses, trucks, and pedestrians using Ultralytics YOLO (`yolov8n.pt` by default for high-speed laptop execution).
- **Persistent Multi-Object Tracking:** Tracks road users across consecutive frames with ByteTrack, maintaining continuous identities, timestamped spatial centers, and bounding-box area histories.
- **Short-Term Trajectory Prediction:** Baseline motion-forecasting engine estimating image-space velocity vectors (`vx`, `vy`) and multi-step future positions across a configurable horizon (1.0 to 3.0 seconds).
- **Collision-Risk Assessment:** Evaluates potential path conflicts within the camera vehicle's forward travel corridor (`ego corridor`), bounding-box scale expansion rate, and image-space time-to-conflict heuristics.
  - Risk Levels: `SAFE`, `LOW RISK`, `MEDIUM RISK`, `HIGH RISK`, and `INSUFFICIENT DATA / UNKNOWN`.
- **Multi-Sensory Warnings:**
  - **Visual:** High-contrast color-coded warning banners (Crimson Red, Amber, Yellow, Green, Gray) with specific conflict explanations.
  - **Audio:** Browser-synthesized spoken warnings (Web Speech API) and audio frequency tones with rate-limiting and configurable warning cooldown to prevent alarm fatigue.
- **Live Telemetry & Trajectory HUD:** Live video streaming overlay with past trajectory trails (green dots), predicted future paths (magenta vectors with arrowheads), forward corridor guidelines, and an active road-user telemetry table.
- **Session Summary & Data Export:** Automatic logging of processed duration, frame rates, inference latencies, class distribution, unique tracking IDs (with re-ID disclaimers), and full event export in both **JSON** and **CSV** formats.
- **Vehicle Camera SDK Abstraction Boundary:** A documented architectural adapter (`VehicleCameraAdapter`) specifying OEM security boundaries, Android Automotive OS Car Service APIs, and hardware requirements.

---

## Project Architecture

```
RoadSense/
├── app.py                         # Application main entrypoint (Streamlit)
├── config.py                      # Centralized configuration parameters and thresholds
├── requirements.txt               # Pinned project dependencies
├── README.md                      # Comprehensive user and architectural documentation
├── .gitignore                     # Git ignore rules
│
├── camera/                        # Camera and video source abstractions
│   ├── base_source.py             # BaseCameraSource abstract class & FrameData structure
│   ├── video_source.py            # VideoFileSource for MP4, AVI, MOV, MKV files
│   ├── webcam_source.py           # WebcamSource for live USB/built-in cameras
│   └── vehicle_camera_adapter.py  # VehicleCameraAdapter boundary for factory OEM SDKs
│
├── models/                        # Computer vision, tracking, and risk models
│   ├── detector.py                # RoadUserDetector wrapping Ultralytics YOLO
│   ├── tracker.py                 # RoadUserTracker using ByteTrack
│   ├── trajectory_predictor.py    # TrajectoryPredictor forecasting image-space paths
│   └── risk_assessor.py           # RiskAssessor conflict evaluation & alert cooldowns
│
├── services/                      # Pipeline orchestration and session management
│   ├── video_processor.py         # VideoProcessor orchestrating CV pipeline & HUD overlays
│   ├── alert_service.py           # AlertService managing browser audio and speech synthesis
│   └── session_service.py         # SessionService tracking metrics and JSON/CSV export
│
├── ui/                            # Streamlit user interface
│   ├── dashboard.py               # Main dashboard layout, control center, and metrics
│   └── components.py              # Dark automotive CSS styling and UI components
│
├── utils/                         # Validation and logging utilities
│   ├── logging_config.py          # Structured application logging
│   └── validation.py              # File path, video format, and configuration validation
│
├── scripts/                       # Helper scripts and synthetic media generators
│   └── generate_sample_video.py   # Generates sample_road.mp4 for testing
│
└── tests/                         # Automated test suite (pytest)
    ├── test_camera_sources.py     # Camera source and fallback tests
    ├── test_risk_assessor.py      # Collision risk and warning cooldown tests
    ├── test_trajectory_predictor.py# Trajectory calculation and direction tests
    ├── test_validation.py         # Media and config validation tests
    └── test_session_service.py    # Metrics aggregation and export tests
```

---

## Supported Environment

- **Operating System:** Windows 10/11, Ubuntu 20.04/22.04 LTS, macOS 12+
- **Python Version:** Python 3.11 (Recommended and verified: 3.11.15)
- **Hardware Requirement:** Normal student laptop CPU (Intel Core i5 / AMD Ryzen 5 or higher); optional CUDA-compatible GPU.

---

## Installation & Setup Instructions

### 1. Clone or Open the Workspace

Navigate to the project root directory:
```bash
cd RoadSense
```

### 2. Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

*(If using `uv`):*
```powershell
uv venv --python 3.11 .venv
.venv\Scripts\activate
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install the required packages using the pinned `requirements.txt`:
```bash
pip install -r requirements.txt
```

*(Or via `uv` for high-speed installation):*
```bash
uv pip install -r requirements.txt
```

---

## Model Setup & Licensing

RoadSense utilizes Ultralytics YOLOv8. The lightweight nano model weights (`yolov8n.pt`, ~6.2 MB) are automatically verified and cached locally on first run.

- **Pretrained weights:** `yolov8n.pt` (automatically cached in the workspace root).
- **Target Road User Classes:** Pedestrian (0), Bicycle (1), Car (2), Motorcycle (3), Bus (5), Truck (7).
- **License Note:** Ultralytics YOLO is distributed under the **AGPL-3.0 License** for open-source evaluation and Enterprise Licenses for commercial vehicle deployment.

To manually pre-cache or verify the model:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

---

## How to Run the Application

Launch the Streamlit web dashboard:

```bash
streamlit run app.py
```

The browser will automatically open at:
```
http://localhost:8501
```

### Running with Included Sample Road Videos
1. Open the application at `http://localhost:8501`.
2. Under **Select Camera / Video Source** in the sidebar, choose:
   - **Real Highway Traffic Video (`highway.mp4`)**: High-definition real-world multi-lane highway traffic featuring multiple cars, trucks, and continuous ByteTrack tracking across 1,000+ frames.
   - **Synthetic Collision Scene (`sample_road.mp4`)**: Controlled collision simulation featuring an approaching vehicle closing rapidly in the forward travel corridor and a crossing pedestrian.
3. Click the **▶️ Start Analysis** button.
4. Observe real-time bounding boxes, trajectory trails, predicted paths, and HUD banners.
5. Click **🔔 Test Sound** or adjust audio settings in the sidebar to verify browser audio warnings.
6. When the video completes or you click **⏹️ Stop**, review the **Session Summary & Safety Event Log** and export data via **JSON** or **CSV** download buttons.

### Analyzing Custom Dashcam Videos
1. Select **Upload Video File** in the sidebar.
2. Drag and drop any supported video file (`.mp4`, `.avi`, `.mov`, `.mkv`, max 500 MB).
3. The file is validated for format, header integrity, and non-empty size.
4. Click **▶️ Start Analysis**.
5. Use **⏸️ Pause** / **▶️ Resume** or **⏹️ Stop** at any time.

### Testing Webcam Mode
1. Ensure a USB webcam or built-in camera is connected.
2. Select **Live Webcam** in the sidebar.
3. Set the device index (default is `0`).
4. Click **▶️ Start Analysis**.
*(If no webcam is attached or permission is denied, a descriptive warning is displayed without crashing).*

---

## Running Automated Tests

RoadSense includes a comprehensive `pytest` suite covering 29 test cases across validation, trajectory forecasting, risk assessment, warning cooldowns, camera abstractions, and end-to-end pipeline execution:

```bash
pytest tests/ -v
```

Expected result:
```
tests/test_camera_sources.py::test_video_file_source_invalid_path PASSED
tests/test_camera_sources.py::test_video_file_source_valid_sample PASSED
tests/test_camera_sources.py::test_webcam_source_invalid_device PASSED
tests/test_camera_sources.py::test_vehicle_camera_adapter_hardware_boundary PASSED
tests/test_camera_sources.py::test_vehicle_camera_adapter_simulation_mode PASSED
tests/test_end_to_end_pipeline.py::test_end_to_end_sample_video_processing PASSED
tests/test_risk_assessor.py::test_risk_approaching_car_in_ego_corridor PASSED
tests/test_risk_assessor.py::test_risk_non_conflicting_lateral_car PASSED
tests/test_risk_assessor.py::test_risk_receding_car_in_corridor PASSED
tests/test_risk_assessor.py::test_risk_vulnerable_pedestrian_in_path PASSED
tests/test_risk_assessor.py::test_risk_insufficient_data_scenario PASSED
tests/test_risk_assessor.py::test_warning_cooldown_behavior PASSED
tests/test_risk_assessor.py::test_empty_scene_assessment PASSED
tests/test_risk_assessor.py::test_ttc_never_presented_as_calibrated_physical_metric PASSED
tests/test_session_service.py::test_session_lifecycle_and_metrics PASSED
tests/test_session_service.py::test_export_json_format PASSED
tests/test_session_service.py::test_export_csv_format PASSED
tests/test_trajectory_predictor.py::test_predict_known_constant_velocity PASSED
tests/test_trajectory_predictor.py::test_predict_insufficient_observations PASSED
tests/test_trajectory_predictor.py::test_predict_irregular_zero_delta_timestamps PASSED
tests/test_trajectory_predictor.py::test_predict_receding_direction PASSED
tests/test_trajectory_predictor.py::test_predict_crossing_direction PASSED
tests/test_validation.py::test_validate_video_file_empty_path PASSED
tests/test_validation.py::test_validate_video_file_nonexistent PASSED
tests/test_validation.py::test_validate_video_file_invalid_extension PASSED
tests/test_validation.py::test_validate_video_file_empty_file PASSED
tests/test_validation.py::test_validate_video_file_valid_mp4 PASSED
tests/test_validation.py::test_validate_config_valid PASSED
tests/test_validation.py::test_validate_config_invalid_thresholds PASSED

============================= 29 passed in 8.82s ==============================
```

---

## Important Safety Boundaries & Technical Limitations

1. **Uncalibrated 2D Image-Space Heuristics:** Pixel displacement and bounding box growth rates are mathematical 2D approximations. Without calibrated camera focal length, optical center, mounting pitch/roll, and vehicle CAN-bus forward velocity, exact 3D physical distances and physical Time-to-Collision (TTC) in seconds cannot be definitively measured.
2. **Experimental Assistance Only:** RoadSense is strictly an experimental Advanced Driver Assistance System (ADAS) demonstration. It does **not** actuate braking, steering, or throttle controls, and must never be treated as an autonomous collision-prevention guarantee.
3. **Tracking Identity Limitations:** Track IDs represent consecutive frame associations maintained by ByteTrack. Real-world occlusions, lighting glare, or sharp camera turns can cause ID reassignments. Total unique tracking IDs should not be conflated with the exact count of unique physical vehicles.
4. **Browser Audio Permissions:** Modern web browsers require at least one user click on the web page before permitting synthetic audio speech or audio oscillator tones. Use the **🔔 Test Sound** button upon page load to activate browser audio permission.

---

## Future Vehicle-Camera Integration Plan

### Target Automotive Environments
1. **Operating Systems:**
   - **Android Automotive OS (AAOS 14+)** running natively on the vehicle Cockpit Domain Controller (CDC).
   - **QNX Neutrino RTOS** or **Automotive Grade Linux (AGL)** running alongside hypervisors.
   - *Note:* Smartphone projection modes (**Android Auto** and **Apple CarPlay**) explicitly restrict camera access for privacy and safety reasons.

2. **Camera Access HAL & Privileges:**
   - Raw factory cameras (ADAS front camera, surround view) are governed by the Android Automotive **Extended View System (EVS)** and `android.car.permission.CAR_CAMERA_SERVICE`.
   - Third-party deployment requires OEM cryptographic signing, ASIL-B / ASIL-D functional safety compliance (ISO 26262), and direct OEM platform authorization.

3. **Sensor Fusion with Vehicle Telemetry:**
   - To convert 2D image coordinates into calibrated 3D world vectors, the processing engine must bind to the Vehicle HAL (VHAL) to read:
     - `VEHICLE_SPEED` (m/s)
     - `STEERING_WHEEL_ANGLE` (degrees)
     - `YAW_RATE` (rad/s)
     - Camera intrinsics matrix ($K$) and extrinsic pose ($R, T$) relative to the vehicle center of gravity.

4. **Hardware Acceleration:**
   - Export PyTorch models to **ONNX / TensorRT / Qualcomm SNPE** for sub-15ms inference latency on automotive chips (e.g., Qualcomm Snapdragon Cockpit 8295 or NVIDIA DRIVE Orin).
   - Real-time decision-making runs strictly on-device without cloud latency.
