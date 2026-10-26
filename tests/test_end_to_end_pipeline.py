"""
End-to-End Pipeline Integration Test
Verifies end-to-end video stream reading, tracking, trajectory forecasting,
collision risk evaluation, and session summary export on a multi-frame video run.
"""
import os
import json
import pandas as pd
import pytest

from config import AppConfig
from camera.video_source import VideoFileSource
from services.video_processor import VideoProcessor
from services.session_service import SessionService
from models.risk_assessor import RiskLevel


def test_end_to_end_sample_video_processing():
    sample_path = "sample_road.mp4"
    assert os.path.exists(sample_path), "sample_road.mp4 must exist for integration testing."

    config = AppConfig()
    processor = VideoProcessor(config)
    session_service = SessionService()

    source = VideoFileSource(sample_path, target_width=480)
    assert source.open(), "Failed to open sample video source."

    processed_frames = 0
    max_test_frames = 25  # Test first 25 frames

    try:
        while processed_frames < max_test_frames:
            frame_data = source.read_frame()
            if frame_data is None:
                break

            result = processor.process_frame(frame_data)
            processed_frames += 1

            # Validate frame output
            assert result.annotated_frame is not None
            assert result.annotated_frame.shape[0] > 0
            assert result.annotated_frame.shape[1] == 480
            assert result.fps_instant > 0
            assert result.latency_ms > 0

            # Record session
            session_service.record_frame(
                frame_timestamp=result.timestamp,
                fps_instant=result.fps_instant,
                latency_ms=result.latency_ms,
                active_track_ids=[t.track_id for t in result.active_tracks],
                active_classes=[t.class_name for t in result.active_tracks]
            )

            for w in result.triggered_warnings:
                session_service.record_warning(
                    timestamp_sec=result.timestamp,
                    track_id=w.track_id,
                    object_class=w.object_class,
                    risk_level=w.risk_level,
                    risk_score=w.risk_score,
                    reason=w.reason,
                    uncalibrated_ttc_sec=w.uncalibrated_ttc_sec
                )

        assert processed_frames == max_test_frames

        session_service.finalize()
        summary = session_service.get_summary_dict()

        assert summary["total_frames_processed"] == max_test_frames
        assert summary["average_fps"] > 0
        assert "tracking_id_disclaimer" in summary

        # Test exports
        json_export = session_service.export_summary_json()
        data = json.loads(json_export)
        assert data["total_frames_processed"] == max_test_frames

        csv_summary = session_service.export_summary_csv()
        assert len(csv_summary) > 20

    finally:
        source.release()
        assert not source.is_opened()
