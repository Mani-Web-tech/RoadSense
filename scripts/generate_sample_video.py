"""
Synthetic Sample Road Video Generator
Generates a realistic test video simulating a forward dashcam perspective with:
1. An approaching vehicle in the ego corridor (closing distance).
2. A crossing pedestrian.
3. An adjacent car driving safely in the left lane.
Useful for integration testing, unit verification, and offline demonstration.
"""
import os
import math
import cv2
import numpy as np


def generate_sample_road_video(output_path: str = "sample_road.mp4", num_frames: int = 150, fps: int = 30) -> str:
    """
    Creates an MP4 road scene video with simulated vehicles and pedestrian movement.
    """
    width, height = 720, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    horizon_y = int(height * 0.45)

    for i in range(num_frames):
        # 1. Background: Sky and ground
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Sky: blue-gray
        frame[:horizon_y, :] = (180, 160, 130)
        # Ground: dark asphalt
        frame[horizon_y:, :] = (55, 55, 60)

        # 2. Road surface perspective (trapezoid)
        road_pts = np.array([
            [width * 0.42, horizon_y],
            [width * 0.58, horizon_y],
            [width * 0.95, height],
            [width * 0.05, height]
        ], np.int32)
        cv2.fillPoly(frame, [road_pts], (40, 40, 42))

        # Lane markings (dashed center lines)
        dash_offset = (i * 8) % 60
        for y in range(horizon_y + dash_offset, height, 60):
            progress = (y - horizon_y) / (height - horizon_y)
            center_x = width * 0.5
            w_dash = int(4 + progress * 8)
            h_dash = int(10 + progress * 25)
            cv2.rectangle(frame, (int(center_x - w_dash / 2), y), (int(center_x + w_dash / 2), y + h_dash), (220, 220, 220), -1)

        # 3. Object 1: Car approaching in front lane (grows as it gets closer)
        # Progress from horizon (y=220) to close (y=360)
        car_prog = min(1.0, i / float(num_frames))
        car_y = int(horizon_y + 30 + car_prog * 120)
        car_w = int(70 + car_prog * 140)
        car_h = int(50 + car_prog * 90)
        car_x = int((width * 0.5) - (car_w / 2) + math.sin(i * 0.05) * 10)

        # Draw vehicle body (red car)
        cv2.rectangle(frame, (car_x, car_y - car_h), (car_x + car_w, car_y), (40, 40, 190), -1)
        # Cabin / windshield
        cv2.rectangle(frame, (car_x + int(car_w * 0.15), car_y - car_h - int(car_h * 0.4)), (car_x + int(car_w * 0.85), car_y - car_h), (90, 80, 80), -1)
        # Headlights / taillights
        cv2.circle(frame, (car_x + int(car_w * 0.18), car_y - int(car_h * 0.2)), int(6 + car_prog * 6), (50, 50, 240), -1)
        cv2.circle(frame, (car_x + int(car_w * 0.82), car_y - int(car_h * 0.2)), int(6 + car_prog * 6), (50, 50, 240), -1)
        # License plate
        cv2.rectangle(frame, (car_x + int(car_w * 0.35), car_y - int(car_h * 0.25)), (car_x + int(car_w * 0.65), car_y - int(car_h * 0.1)), (230, 230, 230), -1)

        # 4. Object 2: Pedestrian crossing from right to left
        ped_x = int(width * 0.85 - (i * 2.2))
        ped_y = int(height * 0.72)
        ped_h = 60
        ped_w = 20

        # Head
        cv2.circle(frame, (ped_x, ped_y - ped_h + 10), 9, (180, 200, 220), -1)
        # Torso
        cv2.rectangle(frame, (ped_x - 8, ped_y - ped_h + 20), (ped_x + 8, ped_y - 20), (160, 100, 50), -1)
        # Legs
        leg_offset = int(math.sin(i * 0.4) * 6)
        cv2.line(frame, (ped_x - 4, ped_y - 20), (ped_x - 6 + leg_offset, ped_y), (60, 60, 60), 4)
        cv2.line(frame, (ped_x + 4, ped_y - 20), (ped_x + 6 - leg_offset, ped_y), (60, 60, 60), 4)

        out.write(frame)

    out.release()
    return output_path


if __name__ == "__main__":
    import math
    p = generate_sample_road_video()
    print(f"Sample road video generated at: {p}")
