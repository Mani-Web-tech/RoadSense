"""
Tests for Risk Assessment Module
Verifies collision risk evaluation under approaching, non-conflicting,
insufficient data, pedestrian prioritization, and warning cooldown scenarios.
"""
import pytest
from models.risk_assessor import RiskAssessor, RiskLevel
from models.tracker import TrackedObject
from models.trajectory_predictor import TrajectoryPrediction
from config import RiskConfig


def _create_test_track(
    track_id: int,
    class_name: str,
    cx: float,
    cy: float,
    w: float = 60.0,
    h: float = 40.0,
    history_len: int = 5
) -> TrackedObject:
    """Helper to create synthetic TrackedObject for deterministic testing."""
    history = [(cx, cy, i * 0.1, w * h) for i in range(history_len)]
    return TrackedObject(
        track_id=track_id,
        class_id=2 if class_name == "car" else 0,
        class_name=class_name,
        confidence=0.88,
        box=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
        center=(cx, cy),
        area=w * h,
        history=history,
        first_seen_timestamp=0.0,
        last_seen_timestamp=0.5,
        consecutive_hits=history_len
    )


def test_risk_approaching_car_in_ego_corridor():
    assessor = RiskAssessor(RiskConfig(warning_cooldown_sec=2.0))
    # Frame 640x480. Ego corridor [0.28, 0.72] -> [179.2, 460.8]
    # Car at center (320, 360) closing rapidly (+vy = 120 px/s)
    track = _create_test_track(track_id=1, class_name="car", cx=320.0, cy=360.0)
    pred = TrajectoryPrediction(
        track_id=1,
        velocity_xy=(0.0, 120.0),
        speed_pixels_sec=120.0,
        heading_deg=90.0,
        motion_direction="Approaching",
        predicted_points=[(320.0, 390.0), (320.0, 420.0), (320.0, 450.0)],
        horizon_sec=1.5,
        area_growth_rate=0.25,
        is_reliable=True
    )

    assessment = assessor.assess_object(track, pred, frame_width=640, frame_height=480, current_timestamp=1.0)
    assert assessment.risk_level == RiskLevel.HIGH
    assert assessment.is_in_ego_corridor
    assert assessment.warning_triggered


def test_risk_non_conflicting_lateral_car():
    assessor = RiskAssessor()
    # Car in far left peripheral lane (cx = 60), moving straight
    track = _create_test_track(track_id=2, class_name="car", cx=60.0, cy=250.0)
    pred = TrajectoryPrediction(
        track_id=2,
        velocity_xy=(0.0, 20.0),
        speed_pixels_sec=20.0,
        heading_deg=90.0,
        motion_direction="Moving",
        predicted_points=[(60.0, 260.0), (60.0, 270.0)],
        horizon_sec=1.5,
        area_growth_rate=0.01,
        is_reliable=True
    )

    assessment = assessor.assess_object(track, pred, frame_width=640, frame_height=480, current_timestamp=1.0)
    assert assessment.risk_level in (RiskLevel.LOW, RiskLevel.SAFE)
    assert not assessment.is_in_ego_corridor
    assert not assessment.warning_triggered


def test_risk_receding_car_in_corridor():
    assessor = RiskAssessor()
    # Car ahead in lane but moving away (-vy = -60 px/s)
    track = _create_test_track(track_id=3, class_name="car", cx=320.0, cy=280.0)
    pred = TrajectoryPrediction(
        track_id=3,
        velocity_xy=(0.0, -60.0),
        speed_pixels_sec=60.0,
        heading_deg=270.0,
        motion_direction="Receding",
        predicted_points=[(320.0, 260.0), (320.0, 240.0)],
        horizon_sec=1.5,
        area_growth_rate=-0.15,
        is_reliable=True
    )

    assessment = assessor.assess_object(track, pred, frame_width=640, frame_height=480, current_timestamp=1.0)
    assert assessment.risk_level == RiskLevel.SAFE
    assert not assessment.warning_triggered


def test_risk_vulnerable_pedestrian_in_path():
    assessor = RiskAssessor()
    # Pedestrian inside ego corridor
    track = _create_test_track(track_id=4, class_name="pedestrian", cx=310.0, cy=320.0)
    pred = TrajectoryPrediction(
        track_id=4,
        velocity_xy=(5.0, 10.0),
        speed_pixels_sec=11.0,
        heading_deg=60.0,
        motion_direction="Crossing Right",
        predicted_points=[(315.0, 325.0), (320.0, 330.0)],
        horizon_sec=1.5,
        area_growth_rate=0.02,
        is_reliable=True
    )

    assessment = assessor.assess_object(track, pred, frame_width=640, frame_height=480, current_timestamp=1.0)
    assert assessment.risk_level == RiskLevel.HIGH
    assert "vulnerable" in assessment.reason.lower()


def test_risk_insufficient_data_scenario():
    assessor = RiskAssessor()
    # Track with only 1 observation (stability score < 0.4)
    track = _create_test_track(track_id=5, class_name="car", cx=320.0, cy=350.0, history_len=1)
    pred = TrajectoryPrediction(
        track_id=5,
        velocity_xy=(0.0, 0.0),
        speed_pixels_sec=0.0,
        heading_deg=0.0,
        motion_direction="Insufficient Observations",
        predicted_points=[],
        horizon_sec=1.5,
        area_growth_rate=0.0,
        is_reliable=False
    )

    assessment = assessor.assess_object(track, pred, frame_width=640, frame_height=480, current_timestamp=1.0)
    assert assessment.risk_level == RiskLevel.INSUFFICIENT_DATA
    assert not assessment.warning_triggered


def test_warning_cooldown_behavior():
    assessor = RiskAssessor(RiskConfig(warning_cooldown_sec=2.0))
    track = _create_test_track(track_id=6, class_name="car", cx=320.0, cy=380.0)
    pred = TrajectoryPrediction(
        track_id=6,
        velocity_xy=(0.0, 100.0),
        speed_pixels_sec=100.0,
        heading_deg=90.0,
        motion_direction="Approaching",
        predicted_points=[(320.0, 410.0)],
        horizon_sec=1.5,
        area_growth_rate=0.22,
        is_reliable=True
    )

    # Frame 1 at t = 1.0s: Alert should trigger
    a1 = assessor.assess_object(track, pred, 640, 480, current_timestamp=1.0)
    assert a1.warning_triggered

    # Frame 2 at t = 1.8s (elapsed 0.8s < 2.0s cooldown): Alert should NOT trigger
    a2 = assessor.assess_object(track, pred, 640, 480, current_timestamp=1.8)
    assert not a2.warning_triggered

    # Frame 3 at t = 3.2s (elapsed 2.2s > 2.0s cooldown): Alert SHOULD trigger again
    a3 = assessor.assess_object(track, pred, 640, 480, current_timestamp=3.2)
    assert a3.warning_triggered


def test_empty_scene_assessment():
    assessor = RiskAssessor()
    summary = assessor.assess_scene(
        active_tracks=[],
        predictions={},
        frame_width=640,
        frame_height=480,
        current_timestamp=0.0
    )
    # Technical Safety Rule: Never report a scene as SAFE merely because no objects were detected
    assert summary.overall_risk_level != RiskLevel.SAFE
    assert summary.overall_risk_level == RiskLevel.NO_TARGETS_DETECTED
    assert summary.active_objects_count == 0
    assert "EMPTY" in summary.scene_reliability
    assert "not guaranteed safe" in summary.highest_risk_reason.lower()


def test_ttc_never_presented_as_calibrated_physical_metric():
    assessor = RiskAssessor()
    track = _create_test_track(track_id=7, class_name="car", cx=320.0, cy=350.0)
    pred = TrajectoryPrediction(
        track_id=7,
        velocity_xy=(0.0, 80.0),
        speed_pixels_sec=80.0,
        heading_deg=90.0,
        motion_direction="Approaching",
        predicted_points=[(320.0, 390.0)],
        horizon_sec=1.5,
        area_growth_rate=0.18,
        is_reliable=True
    )
    assessment = assessor.assess_object(track, pred, 640, 480, current_timestamp=1.0)
    
    # Technical Safety Rule: Closing metric must be explicitly flagged as uncalibrated heuristic
    assert assessment.calibration_status == "UNCALIBRATED_IMAGE_SPACE_HEURISTIC"
    assert "uncalibrated" in assessment.reason.lower()
    # Must not claim certified distance or certified physical TTC
    assert "certified" not in assessment.reason.lower()
