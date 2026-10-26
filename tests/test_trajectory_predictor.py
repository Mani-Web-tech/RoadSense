"""
Tests for Trajectory Prediction Module
Verifies image-space velocity calculation, multi-step projection,
direction classification, and noise/irregular timestamp rejection.
"""
import pytest
from models.trajectory_predictor import TrajectoryPredictor
from config import TrajectoryConfig


def test_predict_known_constant_velocity():
    predictor = TrajectoryPredictor(TrajectoryConfig(prediction_horizon_sec=2.0, min_observations=3))

    # Known linear movement: x moves +20 px/sec, y moves +100 px/sec
    # (cx, cy, timestamp, area)
    history = [
        (100.0, 200.0, 0.0, 1000.0),
        (102.0, 210.0, 0.1, 1020.0),
        (104.0, 220.0, 0.2, 1040.0),
        (106.0, 230.0, 0.3, 1060.0),
    ]

    pred = predictor.predict(track_id=1, history=history)

    assert pred.is_reliable
    assert pred.track_id == 1
    # Check estimated velocity is close to (20, 100)
    vx, vy = pred.velocity_xy
    assert pytest.approx(vx, abs=2.0) == 20.0
    assert pytest.approx(vy, abs=5.0) == 100.0
    assert pred.motion_direction == "Approaching"
    assert len(pred.predicted_points) == 6

    # Verify that future points project forward from the last observation
    last_cx, last_cy = 106.0, 230.0
    p1_x, p1_y = pred.predicted_points[0]
    p_last_x, p_last_y = pred.predicted_points[-1]
    assert p_last_y > p1_y > last_cy


def test_predict_insufficient_observations():
    predictor = TrajectoryPredictor(TrajectoryConfig(min_observations=3))
    # Only 2 observations provided
    history = [
        (100.0, 200.0, 0.0, 1000.0),
        (105.0, 210.0, 0.1, 1000.0)
    ]
    pred = predictor.predict(track_id=2, history=history)
    assert not pred.is_reliable
    assert len(pred.predicted_points) == 0
    assert "insufficient" in pred.motion_direction.lower()


def test_predict_irregular_zero_delta_timestamps():
    predictor = TrajectoryPredictor(TrajectoryConfig(min_observations=3))
    # Identical timestamps (dt = 0)
    history = [
        (100.0, 200.0, 1.0, 1000.0),
        (105.0, 210.0, 1.0, 1000.0),
        (110.0, 220.0, 1.0, 1000.0)
    ]
    pred = predictor.predict(track_id=3, history=history)
    assert not pred.is_reliable
    assert len(pred.predicted_points) == 0


def test_predict_receding_direction():
    predictor = TrajectoryPredictor()
    # Object moving up towards horizon (-y)
    history = [
        (300.0, 300.0, 0.0, 1000.0),
        (300.0, 280.0, 0.2, 900.0),
        (300.0, 260.0, 0.4, 800.0),
        (300.0, 240.0, 0.6, 700.0),
    ]
    pred = predictor.predict(track_id=4, history=history)
    assert pred.is_reliable
    assert pred.motion_direction == "Receding"
    assert pred.velocity_xy[1] < 0


def test_predict_crossing_direction():
    predictor = TrajectoryPredictor()
    # Object moving horizontally right (+x)
    history = [
        (100.0, 300.0, 0.0, 1000.0),
        (130.0, 300.0, 0.2, 1000.0),
        (160.0, 300.0, 0.4, 1000.0),
        (190.0, 300.0, 0.6, 1000.0),
    ]
    pred = predictor.predict(track_id=5, history=history)
    assert pred.is_reliable
    assert pred.motion_direction == "Crossing Right"
