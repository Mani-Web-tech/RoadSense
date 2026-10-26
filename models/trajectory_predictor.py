"""
RoadSense Trajectory Predictor Module
Estimates short-term future image-space movement (1-3 seconds) from timestamped observations.
Uses an exponential moving average velocity baseline with noise rejection.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math
import numpy as np

from config import TrajectoryConfig
from utils.logging_config import logger


@dataclass
class TrajectoryPrediction:
    """Estimated future image-space path for a tracked road user."""
    track_id: int
    velocity_xy: Tuple[float, float]               # (vx, vy) in image pixels per second
    speed_pixels_sec: float                        # Magnitude of velocity
    heading_deg: float                             # Direction angle in degrees (0 = right, 90 = down/approaching)
    motion_direction: str                          # "Approaching", "Receding", "Crossing Left", "Crossing Right", "Stationary"
    predicted_points: List[Tuple[float, float]]    # Future [(px_1, py_1), (px_2, py_2), ...] in image coordinates
    horizon_sec: float                             # Prediction time window
    area_growth_rate: float                        # Fractional change in bbox area per second (dA/dt / A)
    is_reliable: bool                              # True if sufficient quality observations exist
    note: str = "Image-space motion estimate (linear baseline)"


class TrajectoryPredictor:
    """
    Computes image-space velocity and projects short-term future trajectory points.
    Filters out timestamp jitter, irregular sampling, and erratic detections.
    """

    def __init__(self, config: Optional[TrajectoryConfig] = None):
        self.config = config or TrajectoryConfig()

    def predict(
        self,
        track_id: int,
        history: List[Tuple[float, float, float, float]],
        current_time: Optional[float] = None
    ) -> TrajectoryPrediction:
        """
        Calculates velocity and projects future trajectory points.

        Args:
            track_id: ID of the tracked object.
            history: List of (cx, cy, timestamp, area) observations.
            current_time: Current timestamp (optional).

        Returns:
            TrajectoryPrediction object.
        """
        # Minimum observation check
        if len(history) < self.config.min_observations:
            return TrajectoryPrediction(
                track_id=track_id,
                velocity_xy=(0.0, 0.0),
                speed_pixels_sec=0.0,
                heading_deg=0.0,
                motion_direction="Insufficient Observations",
                predicted_points=[],
                horizon_sec=self.config.prediction_horizon_sec,
                area_growth_rate=0.0,
                is_reliable=False,
                note="Awaiting sufficient historical points for trajectory estimation"
            )

        # Filter out observations outside max_history_seconds if timestamp is progressing
        last_obs = history[-1]
        cx_curr, cy_curr, t_curr, area_curr = last_obs

        relevant_obs = [
            obs for obs in history
            if (t_curr - obs[2]) <= self.config.max_history_seconds
        ]
        if len(relevant_obs) < self.config.min_observations:
            relevant_obs = history[-self.config.min_observations:]

        # Calculate image-space velocity using timestamped sequential differences
        vx_list: List[float] = []
        vy_list: List[float] = []
        area_rates: List[float] = []

        for i in range(1, len(relevant_obs)):
            cx0, cy0, t0, a0 = relevant_obs[i - 1]
            cx1, cy1, t1, a1 = relevant_obs[i]
            dt = t1 - t0

            # Reject invalid or zero timestamps
            if dt <= 1e-4:
                continue

            vx = (cx1 - cx0) / dt
            vy = (cy1 - cy0) / dt
            vx_list.append(vx)
            vy_list.append(vy)

            if a0 > 10.0:
                rate = (a1 - a0) / (a0 * dt)
                area_rates.append(rate)

        if not vx_list or not vy_list:
            return TrajectoryPrediction(
                track_id=track_id,
                velocity_xy=(0.0, 0.0),
                speed_pixels_sec=0.0,
                heading_deg=0.0,
                motion_direction="Zero Elapsed Time",
                predicted_points=[],
                horizon_sec=self.config.prediction_horizon_sec,
                area_growth_rate=0.0,
                is_reliable=False,
                note="Invalid or identical observation timestamps"
            )

        # Exponential moving average / weighted velocity smoothing
        alpha = self.config.smoothing_factor
        smoothed_vx = vx_list[0]
        smoothed_vy = vy_list[0]

        for i in range(1, len(vx_list)):
            smoothed_vx = alpha * vx_list[i] + (1.0 - alpha) * smoothed_vx
            smoothed_vy = alpha * vy_list[i] + (1.0 - alpha) * smoothed_vy

        speed = math.hypot(smoothed_vx, smoothed_vy)
        heading_rad = math.atan2(smoothed_vy, smoothed_vx)
        heading_deg = (math.degrees(heading_rad) + 360.0) % 360.0

        avg_area_growth = float(np.mean(area_rates)) if area_rates else 0.0

        # Classify qualitative direction (screen coordinate system: +y is downwards)
        # Objects moving downwards in camera view (+vy) are generally approaching the camera vehicle
        if speed < 15.0:
            direction = "Stationary"
        elif smoothed_vy > 25.0 and abs(smoothed_vx) < smoothed_vy:
            direction = "Approaching"
        elif smoothed_vy < -25.0 and abs(smoothed_vx) < abs(smoothed_vy):
            direction = "Receding"
        elif smoothed_vx > 25.0:
            direction = "Crossing Right"
        elif smoothed_vx < -25.0:
            direction = "Crossing Left"
        else:
            direction = "Moving"

        # Generate future predicted points along smoothed velocity vector
        num_steps = 6
        step_dt = self.config.prediction_horizon_sec / num_steps
        predicted_points: List[Tuple[float, float]] = []

        for step in range(1, num_steps + 1):
            future_t = step * step_dt
            px = cx_curr + (smoothed_vx * future_t)
            py = cy_curr + (smoothed_vy * future_t)
            predicted_points.append((float(px), float(py)))

        return TrajectoryPrediction(
            track_id=track_id,
            velocity_xy=(float(smoothed_vx), float(smoothed_vy)),
            speed_pixels_sec=float(speed),
            heading_deg=float(heading_deg),
            motion_direction=direction,
            predicted_points=predicted_points,
            horizon_sec=self.config.prediction_horizon_sec,
            area_growth_rate=avg_area_growth,
            is_reliable=True,
            note="Linear velocity projection in image coordinates (uncalibrated)"
        )
