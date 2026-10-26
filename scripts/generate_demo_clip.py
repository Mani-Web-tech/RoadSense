"""
generate_demo_clip.py
Runs the full RoadSense AI pipeline on highway.mp4 and exports an annotated
MP4 proof-clip (demo_annotated.mp4) showing real YOLO detections, trajectory
trails, prediction arrows, and risk HUD overlays baked into the video.

Run from the project root:
    .venv\Scripts\python.exe scripts/generate_demo_clip.py
"""
import sys
import os
import time

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from config import AppConfig
from camera.video_source import VideoFileSource
from services.video_processor import VideoProcessor


def generate_demo_clip(
    input_path: str = "highway.mp4",
    output_path: str = "demo_annotated.mp4",
    max_frames: int = 450,          # ~18 seconds @ 25 FPS — enough for a full demo
    target_width: int = 720,
):
    """
    Process `input_path` through the full AI pipeline and save annotated output
    to `output_path`.

    Args:
        input_path:   Source video (highway.mp4 or sample_road.mp4).
        output_path:  Destination annotated MP4.
        max_frames:   Maximum frames to process (keeps file size manageable).
        target_width: Width to resize frames to before AI inference.
    """
    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    print(f"[INFO] Opening source: {input_path}")
    config = AppConfig()
    config.process_frame_width = target_width
    config.display_trails = True
    config.display_predictions = True
    config.display_corridor = True

    source = VideoFileSource(input_path, target_width=target_width)
    if not source.open():
        print("[ERROR] Could not open video source.")
        sys.exit(1)

    meta = source.get_metadata()
    fps = meta.get("fps", 25.0)
    print(f"[INFO] Source metadata: {meta}")

    # Read first frame to get actual output resolution after resize
    first_data = source.read_frame()
    if first_data is None:
        print("[ERROR] Could not read first frame.")
        sys.exit(1)

    # Re-open to reset position
    source.release()
    source = VideoFileSource(input_path, target_width=target_width)
    source.open()

    processor = VideoProcessor(config)

    # Detect frame size from a dry-run on the first frame
    probe_data = source.read_frame()
    if probe_data is None:
        print("[ERROR] Empty stream.")
        sys.exit(1)
    probe_result = processor.process_frame(probe_data)
    h, w = probe_result.annotated_frame.shape[:2]
    print(f"[INFO] Output frame size: {w}x{h} @ {fps:.1f} FPS")

    # Re-open again to start from frame 0
    source.release()
    source = VideoFileSource(input_path, target_width=target_width)
    source.open()
    processor = VideoProcessor(config)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_count = 0
    t_start = time.time()

    print(f"[INFO] Processing up to {max_frames} frames → {output_path}")
    while frame_count < max_frames:
        frame_data = source.read_frame()
        if frame_data is None:
            print("[INFO] End of stream.")
            break

        result = processor.process_frame(frame_data)
        writer.write(result.annotated_frame)
        frame_count += 1

        if frame_count % 50 == 0:
            elapsed = time.time() - t_start
            pct = frame_count / max_frames * 100
            print(f"[INFO] {frame_count}/{max_frames} frames ({pct:.0f}%) — {elapsed:.1f}s elapsed")

    source.release()
    writer.release()

    elapsed = time.time() - t_start
    size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0
    print(f"\n[DONE] Wrote {frame_count} annotated frames to: {output_path}")
    print(f"       File size: {size_mb:.1f} MB | Time: {elapsed:.1f}s")


if __name__ == "__main__":
    generate_demo_clip()
