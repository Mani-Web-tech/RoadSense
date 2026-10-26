"""
Tests for Session Service Module
Verifies session metrics recording, class distribution tracking,
warning event logging, and JSON/CSV export formatting.
"""
import json
import io
import pandas as pd
import pytest
from services.session_service import SessionService


def test_session_lifecycle_and_metrics():
    session = SessionService()

    # Record 3 frames
    session.record_frame(
        frame_timestamp=0.1,
        fps_instant=30.0,
        latency_ms=33.3,
        active_track_ids=[1, 2],
        active_classes=["car", "pedestrian"]
    )
    session.record_frame(
        frame_timestamp=0.2,
        fps_instant=28.0,
        latency_ms=35.0,
        active_track_ids=[2, 3],
        active_classes=["pedestrian", "truck"]
    )
    session.record_frame(
        frame_timestamp=0.3,
        fps_instant=32.0,
        latency_ms=31.0,
        active_track_ids=[1, 3],
        active_classes=["car", "truck"]
    )

    # Record a warning event
    session.record_warning(
        timestamp_sec=0.25,
        track_id=2,
        object_class="pedestrian",
        risk_level="HIGH RISK",
        risk_score=0.95,
        reason="Pedestrian crossing path",
        uncalibrated_ttc_sec=1.1
    )

    session.finalize(total_video_duration=10.0)
    summary = session.get_summary_dict()

    assert summary["total_frames_processed"] == 3
    assert summary["unique_tracking_ids_count"] == 3  # IDs 1, 2, 3
    assert summary["warning_events_count"] == 1
    assert "car" in summary["detections_by_class"]
    assert summary["detections_by_class"]["car"] == 2
    assert summary["detections_by_class"]["pedestrian"] == 2
    assert "tracking_id_disclaimer" in summary


def test_export_json_format():
    session = SessionService()
    session.record_frame(0.1, 30.0, 30.0, [1], ["car"])
    session.record_warning(0.1, 1, "car", "HIGH RISK", 0.9, "Rapid closing", 1.2)
    session.finalize()

    json_str = session.export_summary_json()
    data = json.loads(json_str)

    assert data["total_frames_processed"] == 1
    assert len(data["warning_events"]) == 1
    assert data["warning_events"][0]["track_id"] == 1
    assert data["warning_events"][0]["formatted_time"] == "00:00.10"


def test_export_csv_format():
    session = SessionService()
    session.record_frame(0.5, 25.0, 40.0, [5], ["motorcycle"])
    session.record_warning(0.5, 5, "motorcycle", "MEDIUM RISK", 0.65, "Lane approach", 2.0)
    session.finalize()

    events_csv = session.export_events_csv()
    df_events = pd.read_csv(io.StringIO(events_csv))
    assert len(df_events) == 1
    assert "risk_level" in df_events.columns
    assert df_events.iloc[0]["risk_level"] == "MEDIUM RISK"

    metrics_csv = session.export_summary_csv()
    df_metrics = pd.read_csv(io.StringIO(metrics_csv))
    assert len(df_metrics) > 0
    assert "Metric" in df_metrics.columns
