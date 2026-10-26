"""
generate_collision_scene.py
Creates a dramatic collision-scenario video (`collision_scene.mp4`) by:
  1. Extracting a REAL car image patch from highway.mp4 (so YOLO recognises it)
  2. Animating that car growing from far away into a head-on collision course
  3. Adding a second car cutting in from the left/right at high speed
  4. Blending everything onto a realistic road background taken from highway.mp4

When you open collision_scene.mp4 in the RoadSense app and click Start Analysis:
  → YOLO detects the real car patches
  → ByteTrack tracks them
  → Risk Assessor fires HIGH RISK + CRITICAL COLLISION WARNING + alarm sound

Run from project root:
    .venv\\Scripts\\python.exe scripts/generate_collision_scene.py
"""
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from ultralytics import YOLO


# ─────────────────────────── helpers ──────────────────────────────────────────

def alpha_blend(background: np.ndarray, overlay: np.ndarray, x: int, y: int) -> np.ndarray:
    """Paste `overlay` onto `background` at pixel (x, y) with boundary clipping."""
    bg = background.copy()
    oh, ow = overlay.shape[:2]
    bh, bw = bg.shape[:2]

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + ow, bw), min(y + oh, bh)

    if x2 <= x1 or y2 <= y1:
        return bg

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    bg[y1:y2, x1:x2] = overlay[oy1:oy2, ox1:ox2]
    return bg


def draw_road_background(w: int, h: int, sky_frac: float = 0.38) -> np.ndarray:
    """Draw a perspective road with sky, asphalt, lane markings, and guardrails."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    sky_h = int(h * sky_frac)
    horizon_y = sky_h

    # Sky gradient (blue/grey dusk)
    for row in range(sky_h):
        t = row / sky_h
        r = int(80 + 60 * t)
        g = int(100 + 50 * t)
        b = int(140 + 60 * t)
        frame[row, :] = (b, g, r)

    # Distant mountains / treeline silhouette
    for x in range(w):
        hill_h = int(sky_h * 0.25 + 18 * math.sin(x * 0.015) + 12 * math.cos(x * 0.04))
        hill_y = sky_h - hill_h
        for row in range(max(0, hill_y), sky_h):
            frame[row, x] = (55, 80, 55)

    # Road surface (dark grey asphalt)
    vp_x, vp_y = w // 2, horizon_y          # vanishing point
    road_bottom_half = w * 0.55              # half road width at bottom

    pts = np.array([
        [vp_x - 2, vp_y],
        [vp_x + 2, vp_y],
        [int(vp_x + road_bottom_half), h],
        [int(vp_x - road_bottom_half), h],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [pts], (60, 60, 65))

    # Asphalt texture (subtle noise)
    noise = np.random.randint(0, 12, (h - horizon_y, w, 3), dtype=np.uint8)
    frame[horizon_y:, :] = np.clip(frame[horizon_y:, :].astype(np.int32) + noise - 6, 0, 255).astype(np.uint8)

    # Horizon strip (road edge line)
    cv2.line(frame, (0, horizon_y), (w, horizon_y), (100, 110, 90), 2)

    # Perspective lane dashes (white centre dashes)
    num_dashes = 8
    for i in range(num_dashes):
        t_near = (i + 1) / (num_dashes + 1)
        t_far = (i + 0.5) / (num_dashes + 1)
        y_near = int(horizon_y + (h - horizon_y) * t_near)
        y_far = int(horizon_y + (h - horizon_y) * t_far)
        # Perspective interpolation for x positions
        frac_near = (y_near - horizon_y) / (h - horizon_y)
        frac_far = (y_far - horizon_y) / (h - horizon_y)
        x_near = vp_x
        x_far = vp_x
        thickness = max(1, int(3 * frac_near))
        cv2.line(frame, (x_near, y_near), (x_far, y_far), (220, 220, 210), thickness)

    # Road edge kerb lines (left and right)
    for side in [-1, 1]:
        pts_edge = np.array([
            [vp_x + side * 1, vp_y],
            [vp_x + side * int(road_bottom_half * 0.95), h],
        ], dtype=np.int32)
        cv2.polylines(frame, [pts_edge], False, (200, 200, 180), 2)

    return frame


def smooth_resize(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize image with interpolation chosen by direction."""
    cur_h, cur_w = img.shape[:2]
    interp = cv2.INTER_AREA if (target_w < cur_w or target_h < cur_h) else cv2.INTER_CUBIC
    return cv2.resize(img, (max(1, target_w), max(1, target_h)), interpolation=interp)


def add_motion_blur(img: np.ndarray, intensity: int = 5) -> np.ndarray:
    """Apply horizontal motion blur to simulate fast lateral movement."""
    k = max(3, intensity | 1)
    kernel = np.zeros((k, k))
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


def add_glow_overlay(frame: np.ndarray, x: int, y: int, w: int, h: int,
                     color: tuple = (0, 0, 200), alpha: float = 0.25) -> np.ndarray:
    """Add a semi-transparent danger glow around the threat vehicle bounding box."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 4, y - 4), (x + w + 4, y + h + 4), color, -1)
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)


# ─────────────────────────── car extraction ───────────────────────────────────

def extract_car_patches(video_path: str, model_path: str = "yolov8n.pt",
                        max_patches: int = 3,
                        min_area: int = 8000) -> list:
    """
    Scan highway.mp4 to find the best real car image patches YOLO detects.
    Returns a list of BGR numpy arrays (car crops).
    """
    print("[INFO] Extracting real car patches from highway.mp4 ...")
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    patches = []
    frame_idx = 0

    while len(patches) < max_patches:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 15 != 0:         # sample every 15th frame
            continue

        frame_small = cv2.resize(frame, (720, 405))
        results = model(frame_small, conf=0.45, classes=[2, 7, 5], verbose=False)  # car/truck/bus
        if not results or results[0].boxes is None:
            continue

        boxes = results[0].boxes
        for box in boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            w = x2 - x1
            h = y2 - y1
            if w * h < min_area:
                continue
            crop = frame_small[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            patches.append(crop)
            if len(patches) >= max_patches:
                break

    cap.release()
    if patches:
        print(f"[INFO] Extracted {len(patches)} car patches.")
    else:
        print("[WARN] No car patches found — using procedural car sprite.")
    return patches


def make_procedural_car(w: int, h: int, color_bgr=(50, 80, 200)) -> np.ndarray:
    """
    Draw a realistic enough procedural car sprite that YOLO may detect.
    Uses gradient fills, windshield, headlights, and tyres.
    """
    car = np.zeros((h, w, 3), dtype=np.uint8)
    # Body
    body_y1 = h // 3
    body_y2 = int(h * 0.82)
    cv2.rectangle(car, (int(w * 0.04), body_y1), (int(w * 0.96), body_y2), color_bgr, -1)
    # Roof (smaller, centred)
    roof_y1 = int(h * 0.08)
    roof_y2 = body_y1
    pts_roof = np.array([
        [int(w * 0.22), body_y1],
        [int(w * 0.78), body_y1],
        [int(w * 0.72), roof_y1],
        [int(w * 0.28), roof_y1],
    ], dtype=np.int32)
    cv2.fillPoly(car, [pts_roof], (
        max(0, color_bgr[0] - 30),
        max(0, color_bgr[1] - 30),
        max(0, color_bgr[2] - 30)))
    # Windshield
    pts_wind = np.array([
        [int(w * 0.25), body_y1 + 2],
        [int(w * 0.75), body_y1 + 2],
        [int(w * 0.70), roof_y1 + 4],
        [int(w * 0.30), roof_y1 + 4],
    ], dtype=np.int32)
    cv2.fillPoly(car, [pts_wind], (30, 50, 40))
    # Headlights
    hl_y = int(body_y1 + (body_y2 - body_y1) * 0.15)
    cv2.ellipse(car, (int(w * 0.18), hl_y), (int(w * 0.09), int(h * 0.06)), 0, 0, 360, (220, 230, 255), -1)
    cv2.ellipse(car, (int(w * 0.82), hl_y), (int(w * 0.09), int(h * 0.06)), 0, 0, 360, (220, 230, 255), -1)
    # Grille
    for i in range(3):
        gy = int(body_y1 * 1.0 + (body_y2 - body_y1) * (0.35 + i * 0.12))
        cv2.line(car, (int(w * 0.3), gy), (int(w * 0.7), gy), (20, 20, 20), max(1, h // 30))
    # Wheels
    tyre_y = int(body_y2 + h * 0.02)
    tyre_r = max(4, int(h * 0.14))
    for tx in [int(w * 0.22), int(w * 0.78)]:
        cv2.ellipse(car, (tx, tyre_y), (tyre_r, tyre_r), 0, 0, 360, (20, 20, 20), -1)
        cv2.ellipse(car, (tx, tyre_y), (tyre_r // 2, tyre_r // 2), 0, 0, 360, (60, 60, 60), -1)
    return car


# ─────────────────────────── main generator ───────────────────────────────────

def generate_collision_scene(
    highway_path: str = "highway.mp4",
    output_path: str = "collision_scene.mp4",
    fps: float = 25.0,
    duration_sec: float = 20.0,
    width: int = 720,
    height: int = 405,
):
    """
    Compose and write collision_scene.mp4.

    Timeline (20 seconds):
      0–4 s   : Normal road, one car far ahead (safe, small)
      4–8 s   : Car rapidly approaching center corridor (MEDIUM → HIGH RISK builds)
      8–12 s  : HEAD-ON collision course! Car huge, fills corridor → ALARM FIRES
      12–14 s : Brief clear road (cooldown)
      14–18 s : Second car cuts in from LEFT at speed → new HIGH RISK event
      18–20 s : Clear road, end
    """
    total_frames = int(fps * duration_sec)

    # Extract real car patches so YOLO recognises them
    patches = extract_car_patches(highway_path) if os.path.exists(highway_path) else []

    # Choose front car and side car images
    if len(patches) >= 2:
        front_car_src = patches[0]
        side_car_src  = patches[1]
    elif len(patches) == 1:
        front_car_src = patches[0]
        side_car_src  = patches[0]
    else:
        front_car_src = make_procedural_car(160, 90, color_bgr=(50, 90, 210))
        side_car_src  = make_procedural_car(160, 90, color_bgr=(50, 160, 50))

    # Background: use frames from highway.mp4 as rolling scenery, fallback to drawn road
    bg_frames = []
    if os.path.exists(highway_path):
        cap = cv2.VideoCapture(highway_path)
        for _ in range(total_frames):
            ret, f = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, f = cap.read()
            if ret:
                bg_frames.append(cv2.resize(f, (width, height)))
        cap.release()
        print(f"[INFO] Loaded {len(bg_frames)} background frames from highway.mp4")
    else:
        road_bg = draw_road_background(width, height)
        bg_frames = [road_bg.copy() for _ in range(total_frames)]

    # ── Encode ────────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    vp_x = width // 2      # vanishing point x (centre)
    vp_y = int(height * 0.40)  # approximate horizon y

    print(f"[INFO] Compositing {total_frames} frames → {output_path}")

    for fi in range(total_frames):
        t = fi / fps  # time in seconds
        bg = bg_frames[fi % len(bg_frames)].copy()

        # ── Phase 1 (0–4 s): single car far ahead, small ─────────────────
        if 0 <= t < 4:
            phase_frac = t / 4.0
            car_scale = 0.06 + 0.04 * phase_frac   # small car on horizon
            cw = max(20, int(width * car_scale))
            ch = max(12, int(cw * 0.55))
            car_img = smooth_resize(front_car_src, cw, ch)
            cx = vp_x - cw // 2
            cy = vp_y - ch // 2 + int(10 * phase_frac)
            bg = alpha_blend(bg, car_img, cx, cy)

        # ── Phase 2 (4–8 s): car rapidly approaching ─────────────────────
        elif 4 <= t < 8:
            phase_frac = (t - 4) / 4.0
            car_scale = 0.10 + 0.35 * phase_frac   # grows fast
            cw = max(30, int(width * car_scale))
            ch = max(20, int(cw * 0.55))
            car_img = smooth_resize(front_car_src, cw, ch)
            # Slight drift toward centre top
            cx = vp_x - cw // 2 + int(5 * math.sin(phase_frac * math.pi))
            cy = vp_y - ch // 2 + int(40 * phase_frac)
            bg = add_glow_overlay(bg, cx, cy, cw, ch,
                                  color=(0, 100, 220), alpha=0.12 + 0.15 * phase_frac)
            bg = alpha_blend(bg, car_img, cx, cy)

        # ── Phase 3 (8–12 s): HEAD-ON COLLISION COURSE! ──────────────────
        elif 8 <= t < 12:
            phase_frac = (t - 8) / 4.0
            car_scale = 0.45 + 0.55 * phase_frac   # fills the frame
            cw = max(80, int(width * car_scale))
            ch = max(50, int(cw * 0.55))
            car_img = smooth_resize(front_car_src, cw, ch)
            # Vibrate slightly (collision shake)
            shake = int(6 * math.sin(fi * 0.8)) if phase_frac > 0.5 else 0
            cx = vp_x - cw // 2 + shake
            cy = int(vp_y + (height * 0.3) * phase_frac) - ch // 2
            # Danger red glow intensifies
            red_alpha = 0.20 + 0.40 * phase_frac
            bg = add_glow_overlay(bg, max(0, cx), max(0, cy), min(cw, width), min(ch, height),
                                  color=(0, 0, 200), alpha=red_alpha)
            bg = alpha_blend(bg, car_img, cx, cy)

            # Flash frame red at peak (last 0.5s of this phase)
            if phase_frac > 0.85:
                flash = np.zeros_like(bg)
                flash[:, :] = (0, 0, 180)
                bg = cv2.addWeighted(bg, 0.75, flash, 0.25, 0)

        # ── Phase 4 (12–14 s): cooldown, clear road ──────────────────────
        elif 12 <= t < 14:
            pass   # just background

        # ── Phase 5 (14–18 s): side-cutting car from LEFT ────────────────
        elif 14 <= t < 18:
            phase_frac = (t - 14) / 4.0
            # Car shoots in from the left side at high speed
            car_scale = 0.20 + 0.15 * phase_frac
            cw = max(40, int(width * car_scale))
            ch = max(25, int(cw * 0.55))
            car_img = smooth_resize(side_car_src, cw, ch)
            car_img = add_motion_blur(car_img, intensity=max(3, int(20 * (1 - phase_frac))))

            # Moves from far left into ego corridor
            x_start = -cw
            x_end = int(width * 0.35)
            cx = int(x_start + (x_end - x_start) * phase_frac)
            cy = int(height * 0.55) - ch // 2

            bg = add_glow_overlay(bg, max(0, cx), max(0, cy), min(cw, width - max(0, cx)),
                                  min(ch, height - max(0, cy)),
                                  color=(0, 180, 220), alpha=0.15 + 0.20 * phase_frac)
            bg = alpha_blend(bg, car_img, cx, cy)

        # ── Phase 6 (18–20 s): all clear ─────────────────────────────────
        else:
            pass   # just background

        # Subtle vignette darkening at edges
        vignette = np.ones((height, width), dtype=np.float32)
        cv2.ellipse(vignette, (vp_x, height // 2),
                    (int(width * 0.6), int(height * 0.7)), 0, 0, 360, 0.0, -1)
        vignette = cv2.GaussianBlur(vignette, (0, 0), sigmaX=width // 4)
        vignette = 1.0 - 0.35 * vignette
        for c in range(3):
            bg[:, :, c] = np.clip(bg[:, :, c] * vignette, 0, 255).astype(np.uint8)

        writer.write(bg)

        if fi % 50 == 0:
            print(f"[INFO] Frame {fi}/{total_frames} ({fi/total_frames*100:.0f}%) — t={t:.1f}s")

    writer.release()
    size_mb = os.path.getsize(output_path) / 1024 / 1024 if os.path.exists(output_path) else 0
    print(f"\n[DONE] Wrote collision_scene.mp4 — {size_mb:.1f} MB, {total_frames} frames")
    print("       Open in RoadSense → Upload Video File → select collision_scene.mp4 → Start Analysis")
    print("       Watch for the RED CRITICAL COLLISION WARNING banner and alarm!")


if __name__ == "__main__":
    generate_collision_scene()
