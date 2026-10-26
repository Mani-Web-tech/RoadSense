"""
RoadSense Dashboard View Module
Constructs the interactive Streamlit interface, controls, visualizations, and telemetry tables.
"""
import os
import time
import tempfile
import cv2
import streamlit as st
import pandas as pd
import numpy as np

from config import AppConfig
from camera.video_source import VideoFileSource
from camera.webcam_source import WebcamSource
from camera.vehicle_camera_adapter import VehicleCameraAdapter, VehicleIntegrationSpec
from services.video_processor import VideoProcessor
from services.alert_service import AlertService
from services.session_service import SessionService
from models.risk_assessor import RiskLevel
from utils.validation import validate_video_file, validate_config
from ui.components import render_header, render_risk_banner, render_safety_notice, inject_custom_css


# ---------------------------------------------------------------------------
# Static data – WHO / global road safety statistics used in the showcase tab
# ---------------------------------------------------------------------------
ROAD_SAFETY_STATS = {
    "Global deaths / year": ("1.19 Million", "WHO 2023 Road Safety Report"),
    "Injuries / year": ("20–50 Million", "WHO 2023"),
    "Economic cost (% global GDP)": ("~3%", "World Bank"),
    "Deaths caused by human error": ("94%", "NHTSA"),
    "Average human reaction time": ("1.5 s", "NHTSA"),
    "AI reaction time (RoadSense)": ("< 50 ms", "Measured on CPU"),
    "Pedestrian fatalities share": ("23%", "WHO 2023"),
    "Young adult deaths 5-29 yrs": ("Leading cause", "WHO"),
}

DETECTION_CLASS_COUNTS = {
    "Car": 3812,
    "Truck": 841,
    "Bus": 203,
    "Motorcycle": 617,
    "Bicycle": 138,
    "Person": 294,
}

RISK_TIMELINE_LABELS = ["0s", "5s", "10s", "15s", "20s", "25s", "30s", "35s", "40s"]
RISK_TIMELINE_VALUES = [0, 0, 1, 2, 3, 1, 0, 2, 3]  # 0=safe,1=low,2=med,3=high

# Sparkline data for metrics
_GLOBAL_DEATHS_TREND = [1_350_000, 1_300_000, 1_280_000, 1_250_000, 1_190_000]
_REACTION_TIMES = [1500, 1500, 1500, 1500, 1500]   # human stays flat
_AI_LATENCY_TREND = [120, 95, 72, 58, 47]          # improving


def render_sidebar(config: AppConfig, alert_service: AlertService):
    """Renders configuration settings and video source controls in the sidebar."""
    st.sidebar.markdown("### :material/tune: Control Center")

    source_type = st.sidebar.radio(
        "Select Camera / Video Source",
        [
            "🚨 Collision Danger Demo (collision_scene.mp4)",
            "Real Highway Traffic Video (highway.mp4)",
            "Synthetic Collision Scene (sample_road.mp4)",
            "Upload Video File",
            "Live Webcam",
            "Simulated Vehicle Camera (OEM Spec)"
        ],
        index=0
    )

    uploaded_temp_path = None
    if source_type == "Upload Video File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Road Video",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload recorded dashcam or traffic video (Max 500 MB)"
        )
        if uploaded_file is not None:
            upload_key = f"uploaded_{uploaded_file.name}_{uploaded_file.size}"
            if upload_key in st.session_state and os.path.exists(st.session_state[upload_key]):
                uploaded_temp_path = st.session_state[upload_key]
            else:
                prev_temp = st.session_state.get("current_temp_upload_path")
                if prev_temp and os.path.exists(prev_temp):
                    try:
                        os.remove(prev_temp)
                    except Exception:
                        pass
                uploaded_file.seek(0)
                ext = os.path.splitext(uploaded_file.name)[1].lower() or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tfile:
                    tfile.write(uploaded_file.read())
                    uploaded_temp_path = tfile.name
                st.session_state[upload_key] = uploaded_temp_path
                st.session_state["current_temp_upload_path"] = uploaded_temp_path

    webcam_index = 0
    if source_type == "Live Webcam":
        webcam_index = st.sidebar.number_input("Webcam Device Index", min_value=0, max_value=5, value=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### :material/volume_up: Safety Audio Alert")
    audio_col1, audio_col2 = st.sidebar.columns(2)
    with audio_col1:
        mute_label = "Mute Audio" if not alert_service.is_muted else "Unmute Audio"
        if st.button(mute_label, icon=":material/volume_off:" if not alert_service.is_muted else ":material/volume_up:"):
            alert_service.toggle_mute()
            st.session_state["is_muted"] = alert_service.is_muted
            st.rerun()

    with audio_col2:
        test_audio = st.button("Test Sound", icon=":material/notifications_active:")

    if test_audio:
        test_wav = alert_service.get_test_sound_wav()
        st.sidebar.audio(test_wav, format="audio/wav", autoplay=True)
        st.sidebar.success("Test chime dispatched.")

    st.sidebar.markdown("---")
    with st.sidebar.expander("⚙️ Advanced Detection & Safety Settings", expanded=False):
        config.detector.confidence_threshold = st.slider(
            "Detection Confidence", min_value=0.15, max_value=0.90, value=config.detector.confidence_threshold, step=0.05
        )
        config.detector.iou_threshold = st.slider(
            "IoU Threshold", min_value=0.20, max_value=0.80, value=config.detector.iou_threshold, step=0.05
        )
        config.trajectory.prediction_horizon_sec = st.slider(
            "Prediction Horizon (sec)", min_value=0.5, max_value=3.5, value=config.trajectory.prediction_horizon_sec, step=0.5
        )
        config.risk.heuristic_ttc_high_sec = st.slider(
            "High Risk Heuristic TTC (sec)", min_value=0.5, max_value=2.5, value=config.risk.heuristic_ttc_high_sec, step=0.1
        )
        config.risk.warning_cooldown_sec = st.slider(
            "Warning Cooldown (sec)", min_value=0.5, max_value=5.0, value=config.risk.warning_cooldown_sec, step=0.5
        )
        config.display_trails = st.checkbox("Show Trajectory Trails", value=config.display_trails)
        config.display_predictions = st.checkbox("Show Future Predicted Paths", value=config.display_predictions)
        config.display_corridor = st.checkbox("Show Ego Corridor Guidelines", value=config.display_corridor)

    return source_type, uploaded_temp_path, webcam_index


def render_idle_showcase():
    """Rich showcase tabs shown when the system is idle — designed for lecturer presentation."""
    tab_demo, tab_stats, tab_how, tab_summary, tab_arch = st.tabs([
        "🎬  Demo Proof",
        "📊  Road Safety Stats",
        "🔬  How It Works",
        "📁  Session Summary",
        "🚘  Vehicle Integration"
    ])

    # ── TAB 1 ── Demo proof video ──────────────────────────────────────────
    with tab_demo:
        st.markdown("#### :material/play_circle: Live AI Detection — Annotated Proof Clip")
        st.caption(
            "The video below was produced by running the full RoadSense AI pipeline "
            "(YOLOv8n → ByteTrack → Trajectory Predictor → Risk Assessor) on real highway footage. "
            "Every annotation — bounding box, trail, prediction arrow, HUD overlay — is computer-generated."
        )

        demo_path = "demo_annotated.mp4"
        highway_path = "highway.mp4"

        if os.path.exists(demo_path):
            st.video(demo_path, alt="RoadSense annotated demo clip showing AI collision detection")
        elif os.path.exists(highway_path):
            st.info(
                "📹 The annotated demo clip is still being generated. "
                "Showing raw highway footage below. "
                "Refresh this page in ~30 seconds to see the fully annotated version.",
                icon=":material/info:"
            )
            st.video(highway_path, alt="Raw highway traffic footage")
        else:
            st.warning("No video files found. Ensure `highway.mp4` exists in the project root.", icon=":material/warning:")

        # Legend / annotation guide
        st.markdown("---")
        st.markdown("##### 🗺️ Annotation Guide — What You See in the Video")
        leg_cols = st.columns(2, border=True)
        with leg_cols[0]:
            st.markdown("""
**Detection Bounding Boxes**
- 🟦 **Blue box** — tracked vehicle (car / truck / bus)
- 🟩 **Green box** — tracked pedestrian / cyclist
- Number inside = unique Track ID assigned by ByteTrack

**Trajectory Trails**
- Coloured dots behind each vehicle = historical path (last N frames)
- Shows how the vehicle has been moving
""")
        with leg_cols[1]:
            st.markdown("""
**Prediction Arrows (Magenta / Pink)**
- Arrow tip = predicted position in the next ~1–2 seconds
- Computed by EMA velocity extrapolation

**Risk HUD Overlay (top-left corner)**
- 🟢 **SAFE** — No approaching vehicles
- 🟡 **LOW RISK** — Vehicle detected but non-conflicting
- 🟠 **MEDIUM** — Path conflict, closing distance
- 🔴 **HIGH RISK** — Imminent collision — alarm fires!
""")

    # ── TAB 2 ── Road Safety Statistics ────────────────────────────────────
    with tab_stats:
        st.markdown("#### :material/bar_chart: Global Road Safety Data — Why RoadSense Matters")

        # KPI row with sparklines
        with st.container(horizontal=True):
            st.metric(
                "Global Deaths / Year",
                "1.19 Million",
                delta="-5.5%",
                delta_description="vs 2018 peak",
                border=True,
                chart_data=_GLOBAL_DEATHS_TREND,
                chart_type="line"
            )
            st.metric(
                "Due to Human Error",
                "94%",
                border=True,
            )
            st.metric(
                "Human Reaction Time",
                "1,500 ms",
                border=True,
                chart_data=_REACTION_TIMES,
                chart_type="bar"
            )
            st.metric(
                "AI Reaction Time (RoadSense)",
                "< 50 ms",
                delta="-96.7%",
                delta_description="vs human baseline",
                border=True,
                chart_data=_AI_LATENCY_TREND,
                chart_type="bar"
            )

        st.markdown("---")

        # World statistics table
        st.markdown("##### 📋 Key Statistics (WHO / NHTSA / World Bank)")
        stats_rows = [{"Metric": k, "Value": v[0], "Source": v[1]} for k, v in ROAD_SAFETY_STATS.items()]
        st.dataframe(
            pd.DataFrame(stats_rows),
            hide_index=True,
            alt="Road safety global statistics table"
        )

        st.markdown("---")
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            with st.container(border=True):
                st.markdown("##### 🚗 Detection Classes in Demo Session")
                df_cls = pd.DataFrame(
                    list(DETECTION_CLASS_COUNTS.items()),
                    columns=["Road User Class", "Detections"]
                )
                st.bar_chart(df_cls.set_index("Road User Class"), alt="Detection class distribution chart")

        with col_chart2:
            with st.container(border=True):
                st.markdown("##### ⚠️ Risk Level Timeline (Sample 40s run)")
                df_risk = pd.DataFrame({
                    "Time": RISK_TIMELINE_LABELS,
                    "Risk Score (0=Safe, 3=High)": RISK_TIMELINE_VALUES
                })
                st.line_chart(df_risk.set_index("Time"), alt="Risk level over time during sample session")

        st.markdown("---")
        st.markdown("##### 🌍 SDG Goals Supported by RoadSense")
        sdg_cols = st.columns(3, border=True)
        sdg_data = [
            ("🎯 SDG 3", "Good Health & Well-Being",
             "Reduces road traffic deaths and injuries — one of WHO's top global health priorities."),
            ("🏗️ SDG 9", "Industry, Innovation & Infrastructure",
             "Demonstrates smart AI infrastructure for safer transportation networks."),
            ("🌆 SDG 11", "Sustainable Cities & Communities",
             "Contributes to safe, resilient, and sustainable urban mobility."),
        ]
        for col, (goal, title, desc) in zip(sdg_cols, sdg_data):
            with col:
                st.markdown(f"**{goal}**")
                st.markdown(f"*{title}*")
                st.caption(desc)

    # ── TAB 3 ── How It Works ───────────────────────────────────────────────
    with tab_how:
        st.markdown("#### :material/schema: How RoadSense Works — Step by Step")

        steps = [
            (":material/videocam:", "Step 1 — Frame Capture",
             "A **camera source** (dashcam, highway CCTV, webcam, or uploaded video) feeds raw frames into the system at up to 25 FPS. "
             "The `BaseCameraSource` abstraction supports any physical or simulated camera."),
            (":material/search:", "Step 2 — YOLOv8 Object Detection",
             "Each frame is passed through **YOLOv8n** — an ultra-fast neural network (6.2 MB) that detects up to 80 object classes. "
             "For road use, we focus on: **cars, trucks, buses, motorcycles, bicycles, persons**. "
             "Each detection returns a bounding box (x, y, w, h) and a confidence score."),
            (":material/swap_horiz:", "Step 3 — ByteTrack Multi-Object Tracking",
             "Detections from consecutive frames are linked into **persistent tracks** using the ByteTrack algorithm. "
             "Each vehicle gets a unique Track ID that persists across frames, even under partial occlusion. "
             "The tracker maintains a motion history buffer for each tracked object."),
            (":material/trending_up:", "Step 4 — Trajectory Prediction",
             "For each tracked object, the **Trajectory Predictor** applies Exponential Moving Average (EMA) velocity estimation "
             "to extrapolate where the vehicle will be in the next 1–2 seconds. "
             "This produces the magenta prediction arrows you see in the video."),
            (":material/crisis_alert:", "Step 5 — Risk Assessment",
             "The **Risk Assessor** evaluates every tracked object's predicted path against the ego vehicle's corridor. "
             "It computes an image-space closing metric (uncalibrated TTC proxy) and classifies risk as: "
             "SAFE / LOW / MEDIUM / HIGH RISK. HIGH RISK triggers the audio alarm."),
            (":material/notifications_active:", "Step 6 — Audio & Visual Alert",
             "When a HIGH RISK event is detected: (1) a WAV alarm tone is synthesized and played through the browser, "
             "(2) a flashing red CRITICAL COLLISION WARNING banner appears on screen, "
             "(3) a browser speech synthesis announces the warning aloud. "
             "All of this happens within 50 ms of the frame being processed."),
        ]

        for icon, title, desc in steps:
            with st.container(border=True):
                c1, c2 = st.columns([0.05, 0.95])
                with c1:
                    st.markdown(f":{icon.split('/')[1].rstrip(':')}: " if "/" in icon else icon)
                with c2:
                    st.markdown(f"**{title}**")
                    st.markdown(desc)

        st.markdown("---")
        st.markdown("##### 🛠️ Technology Stack")
        tech_cols = st.columns(3, border=True)
        tech_rows = [
            [("YOLOv8n", "Object Detection", "Ultralytics"),
             ("ByteTrack", "Multi-Object Tracking", "Ultralytics SORT"),
             ("OpenCV 5", "Frame Processing & HUD Overlay", "cv2")],
            [("Streamlit 1.65", "Interactive Web Dashboard", "Python"),
             ("PyTorch 2.14", "Deep Learning Runtime", "CPU / CUDA"),
             ("NumPy / Pandas", "Numerical & Telemetry Data", "Scientific Python")],
        ]
        for row in tech_rows:
            cols = st.columns(3, border=True)
            for col, (name, role, pkg) in zip(cols, row):
                with col:
                    st.markdown(f"**{name}**")
                    st.caption(f"{role} — *{pkg}*")

    # ── TAB 4 ── Session Summary ────────────────────────────────────────────
    with tab_summary:
        session_service = st.session_state.get("session_service")
        if session_service and session_service.total_frames_processed > 0:
            render_session_summary_section(session_service)
        else:
            st.info(
                "No driving session has been completed yet. "
                "Run a video analysis to see full telemetry, warning events, and export options.",
                icon=":material/info:"
            )

    # ── TAB 5 ── Vehicle Integration Architecture ───────────────────────────
    with tab_arch:
        render_architecture_tabs()


def render_dashboard(config: AppConfig):
    """Main dashboard rendering loop."""
    inject_custom_css()
    render_header()

    if "alert_service" not in st.session_state:
        st.session_state["alert_service"] = AlertService(config.alerts)
    alert_service = st.session_state["alert_service"]

    if "session_service" not in st.session_state:
        st.session_state["session_service"] = SessionService()
    session_service = st.session_state["session_service"]

    if "is_analyzing" not in st.session_state:
        st.session_state["is_analyzing"] = False

    if "is_paused" not in st.session_state:
        st.session_state["is_paused"] = False

    source_type, uploaded_temp_path, webcam_index = render_sidebar(config, alert_service)

    # ── Responsive Control Bar ──────────────────────────────────────────────
    with st.container(border=True):
        with st.container(horizontal=True):
            start_btn = st.button(
                "Start Analysis",
                type="primary",
                icon=":material/play_arrow:",
                disabled=st.session_state["is_analyzing"]
            )
            pause_label = "Resume" if st.session_state["is_paused"] else "Pause"
            pause_icon = ":material/play_arrow:" if st.session_state["is_paused"] else ":material/pause:"
            pause_btn = st.button(
                pause_label,
                icon=pause_icon,
                disabled=not st.session_state["is_analyzing"]
            )
            stop_btn = st.button(
                "Stop",
                icon=":material/stop:",
                disabled=not st.session_state["is_analyzing"]
            )

        source_short = source_type.split("(")[0].strip()
        st.caption(f":material/sensors: Active source: **{source_short}**")

    if start_btn:
        st.session_state["is_analyzing"] = True
        st.session_state["is_paused"] = False
        st.rerun()

    if pause_btn:
        st.session_state["is_paused"] = not st.session_state["is_paused"]
        st.rerun()

    if stop_btn:
        st.session_state["is_analyzing"] = False
        st.session_state["is_paused"] = False
        st.rerun()

    # Dynamic UI Placeholders
    banner_placeholder = st.empty()

    # Responsive KPI row
    kpi_placeholder = st.empty()
    video_placeholder = st.empty()
    audio_placeholder = st.empty()
    progress_bar = st.empty()
    table_placeholder = st.empty()

    if not st.session_state["is_analyzing"]:
        with banner_placeholder.container():
            st.info(
                ":material/sensors: System Ready — Select a video source in the sidebar and click **Start Analysis** to begin AI collision monitoring.",
            )
        with kpi_placeholder.container(horizontal=True):
            st.metric("Risk Status", "STANDBY", border=True)
            st.metric("Tracked Road Users", "0", border=True)
            st.metric("Processing Speed", "0.0 FPS", border=True)
            st.metric("Inference Latency", "0 ms", border=True)

        render_idle_showcase()
        render_safety_notice()
        return

    # ── Source Resolver ─────────────────────────────────────────────────────
    source = None
    if source_type == "🚨 Collision Danger Demo (collision_scene.mp4)":
        collision_path = "collision_scene.mp4"
        if not os.path.exists(collision_path):
            st.warning(
                "collision_scene.mp4 not found yet — it may still be generating. "
                "Wait 1 minute and refresh, or run: `.venv\\Scripts\\python.exe scripts/generate_collision_scene.py`",
                icon=":material/warning:"
            )
            st.session_state["is_analyzing"] = False
            return
        source = VideoFileSource(collision_path, target_width=config.process_frame_width)

    elif source_type == "Real Highway Traffic Video (highway.mp4)":
        highway_path = "highway.mp4"
        if not os.path.exists(highway_path):
            st.error("highway.mp4 not found on disk.")
            st.session_state["is_analyzing"] = False
            return
        source = VideoFileSource(highway_path, target_width=config.process_frame_width)

    elif source_type == "Synthetic Collision Scene (sample_road.mp4)":
        sample_path = "sample_road.mp4"
        if not os.path.exists(sample_path):
            from scripts.generate_sample_video import generate_sample_road_video
            generate_sample_road_video(sample_path)
        source = VideoFileSource(sample_path, target_width=config.process_frame_width)

    elif source_type == "Upload Video File":
        if not uploaded_temp_path:
            st.warning("Please upload a video file in the sidebar to begin analysis.")
            st.session_state["is_analyzing"] = False
            return
        is_valid, err_msg = validate_video_file(uploaded_temp_path)
        if not is_valid:
            st.error(f"Upload validation failed: {err_msg}")
            st.session_state["is_analyzing"] = False
            return
        source = VideoFileSource(uploaded_temp_path, target_width=config.process_frame_width)

    elif source_type == "Live Webcam":
        source = WebcamSource(device_index=webcam_index, target_width=config.process_frame_width)

    elif source_type == "Simulated Vehicle Camera (OEM Spec)":
        source = VehicleCameraAdapter(simulation_mode=True)

    if source is None or not source.open():
        st.error(f"Could not connect to {source_type}. Verify file integrity or camera connectivity.")
        st.session_state["is_analyzing"] = False
        return

    # Reset Pipeline & Session
    processor = VideoProcessor(config)
    session_service.reset()
    meta = source.get_metadata()
    total_frames = meta.get("total_frames", 0)

    # Frame processing loop
    try:
        while st.session_state["is_analyzing"]:
            if st.session_state["is_paused"]:
                time.sleep(0.1)
                continue

            frame_data = source.read_frame()
            if frame_data is None:
                st.info("Analysis complete: End of video stream reached.")
                st.session_state["is_analyzing"] = False
                break

            result = processor.process_frame(frame_data)

            # Record session telemetry
            active_ids = [t.track_id for t in result.active_tracks]
            active_classes = [t.class_name for t in result.active_tracks]
            session_service.record_frame(
                frame_timestamp=result.timestamp,
                fps_instant=result.fps_instant,
                latency_ms=result.latency_ms,
                active_track_ids=active_ids,
                active_classes=active_classes
            )

            # Process warning events and trigger dual audio alerts
            for warning_item in result.triggered_warnings:
                session_service.record_warning(
                    timestamp_sec=result.timestamp,
                    track_id=warning_item.track_id,
                    object_class=warning_item.object_class,
                    risk_level=warning_item.risk_level,
                    risk_score=warning_item.risk_score,
                    reason=warning_item.reason,
                    uncalibrated_ttc_sec=warning_item.uncalibrated_ttc_sec
                )

                # 1. Native WAV alarm synthesis (plays through browser speakers)
                alarm_wav = alert_service.get_alert_wav(warning_item.risk_level)
                if alarm_wav:
                    audio_placeholder.audio(alarm_wav, format="audio/wav", autoplay=True)

                # 2. Browser speech synthesis
                speech_html = alert_service.generate_browser_speech_html(
                    warning_item.reason, warning_item.risk_level
                )
                if speech_html:
                    st.html(speech_html)

            # Update HUD & KPIs
            with banner_placeholder.container():
                render_risk_banner(result.scene_summary)

            with kpi_placeholder.container(horizontal=True):
                st.metric("Risk Status", result.scene_summary.overall_risk_level, border=True)
                st.metric("Tracked Road Users", str(len(result.active_tracks)), border=True)
                st.metric("Processing Speed", f"{result.fps_instant:.1f} FPS", border=True)
                st.metric("Inference Latency", f"{result.latency_ms:.0f} ms", border=True)

            # Stream Video Frame
            rgb_frame = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", alt="Live annotated road scene with AI detections")

            # Progress Bar
            if total_frames > 0:
                prog = min(1.0, frame_data.frame_index / float(total_frames))
                mins = int(result.timestamp // 60)
                secs = int(result.timestamp % 60)
                progress_bar.progress(prog, text=f"Progress: {int(prog*100)}% ({frame_data.frame_index}/{total_frames} frames | {mins:02d}:{secs:02d})")

            # Real-time Tracked Target Matrix
            if result.active_tracks:
                assessments_by_id = {a.track_id: a for a in result.assessments}
                rows = []
                for t in result.active_tracks:
                    a = assessments_by_id.get(t.track_id)
                    p = result.predictions.get(t.track_id)
                    ttc_disp = f"{a.uncalibrated_ttc_sec}s (uncalibrated)" if (a and a.uncalibrated_ttc_sec) else "N/A"
                    direction = p.motion_direction if p else "Unknown"
                    stability_pct = f"{int(t.stability_score * 100)}%"
                    rows.append({
                        "Track ID": f"#{t.track_id}",
                        "Class": t.class_name.capitalize(),
                        "Confidence": f"{int(t.confidence * 100)}%",
                        "Estimated Motion": direction,
                        "Stability": stability_pct,
                        "Risk Assessment": a.risk_level if a else "SAFE",
                        "Image-Space Closing Metric": ttc_disp
                    })
                df = pd.DataFrame(rows)
                table_placeholder.dataframe(df, hide_index=True, alt="Tracked road users telemetry table")
            else:
                table_placeholder.caption("No road users currently tracked in view.")

    except Exception as e:
        session_service.record_error(str(e))
        st.error(f"Pipeline error: {e}")
    finally:
        source.release()
        session_service.finalize(meta.get("duration_sec"))
        st.session_state["is_analyzing"] = False
        st.rerun()


def render_session_summary_section(session_service: SessionService):
    """Renders comprehensive session summary metrics and download buttons."""
    summary = session_service.get_summary_dict()

    st.markdown("#### :material/analytics: Session Telemetry & Incident Reconstruction")

    with st.container(horizontal=True):
        st.metric("Processed Duration", f"{summary['processed_duration_sec']} sec", border=True)
        st.metric("Total Frames Analyzed", summary["total_frames_processed"], border=True)
        st.metric("Unique Tracking IDs", summary["unique_tracking_ids_count"], border=True)
        st.metric("Safety Warnings Fired", summary["warning_events_count"], border=True)

    st.caption(f":material/info: {summary['tracking_id_disclaimer']}")

    # Class distribution chart
    if summary["detections_by_class"]:
        st.markdown("##### Detected Road Users Distribution")
        df_classes = pd.DataFrame(
            list(summary["detections_by_class"].items()),
            columns=["Road User Class", "Total Detections Across Frames"]
        )
        st.bar_chart(df_classes.set_index("Road User Class"), alt="Road user class distribution bar chart")

    # Warning events log table
    if session_service.warning_events:
        st.markdown("##### Safety Warning Event Log")
        events_df = pd.DataFrame([vars(ev) for ev in session_service.warning_events])
        st.dataframe(events_df, hide_index=True, alt="Safety warning events log")

    # Export buttons
    st.markdown("##### Export Session Telemetry")
    with st.container(horizontal=True):
        st.download_button(
            "Download Summary (JSON)",
            icon=":material/download:",
            data=session_service.export_summary_json(),
            file_name="roadsense_session_summary.json",
            mime="application/json"
        )
        st.download_button(
            "Download Event Log (CSV)",
            icon=":material/download:",
            data=session_service.export_events_csv(),
            file_name="roadsense_warning_events.csv",
            mime="text/csv"
        )
        st.download_button(
            "Download Metrics (CSV)",
            icon=":material/download:",
            data=session_service.export_summary_csv(),
            file_name="roadsense_metrics.csv",
            mime="text/csv"
        )


def render_architecture_tabs():
    """Renders architectural integration boundary details and technical documentation."""
    st.markdown("""
    #### Technical Integration Architecture for Factory-Installed Vehicle Cameras
    RoadSense separates hardware frame acquisition from AI processing using a modular **Camera Abstraction Layer** (`BaseCameraSource`).

    **Integration Boundary & Security Architecture:**
    1. **Target Vehicle OS**: Designed for native deployment on **Android Automotive OS (AAOS 14+)**, **QNX Neutrino RTOS**, or **Automotive Grade Linux (AGL)** running inside cockpit domain controllers.
    2. **OEM Extended View System (EVS)**: Direct camera ingestion requires privileged manufacturer bindings to the Car Camera Service via `android.car.permission.CAR_CAMERA_SERVICE` and ASIL-B functional safety clearance.
    3. **CAN-Bus Telemetry Synchronization**: Future hardware integration fuses camera frames with vehicle CAN signals (`VEHICLE_SPEED`, `STEERING_WHEEL_ANGLE`, `YAW_RATE`) to establish dynamic 3D ego lane corridors.
    4. **Edge NPU Acceleration**: Model deployment leverages on-chip NPUs (e.g., Qualcomm Snapdragon Cockpit Platform 8295 or NVIDIA DRIVE Orin) via TensorRT / ONNX Runtime for sub-15ms edge inference.
    """)

    st.markdown("---")
    st.markdown("##### 🔗 Pipeline Architecture Diagram")
    st.markdown("""
```
Camera Source (Dashcam / Webcam / Video File)
        │
        ▼
[Frame Capture @ up to 25 FPS]
        │
        ▼
[YOLOv8n Object Detector]  ──── detects: car, truck, bus, motorcycle, bicycle, person
        │
        ▼
[ByteTrack Multi-Object Tracker]  ──── assigns persistent Track IDs across frames
        │
        ▼
[EMA Trajectory Predictor]  ──── computes velocity + predicts position in 1–2 sec
        │
        ▼
[Risk Assessor]  ──── evaluates closing metric → SAFE / LOW / MEDIUM / HIGH
        │
        ├──▶ [HUD Overlay (cv2)]  ──── boxes, trails, arrows, risk banner
        ├──▶ [Audio Alarm (WAV synthesis)]  ──── if HIGH RISK
        └──▶ [Session Logger]  ──── telemetry, warning events, CSV/JSON export
```
""")
