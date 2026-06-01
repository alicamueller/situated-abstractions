from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from PIL import Image, ImageDraw

INPUT_DIR = Path("images_nobg")
OUTPUT_DIR = Path("pose_renders")
MODEL_PATH = Path("models/pose_landmarker_full.task")
OUTPUT_DIR.mkdir(exist_ok=True)

# Current MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Standard MediaPipe pose skeleton connections
CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (0, 11), (0, 12),
    (0, 23), (0, 24),
    (11, 22), (12, 21),
    (23, 24)
]

def draw_pose_on_black(landmarks, size):
    w, h = size
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    draw = ImageDraw.Draw(canvas)

    pts = []
    for lm in landmarks:
        x = lm.x * w
        y = lm.y * h
        pts.append((x, y))

    # lines
    for a, b in CONNECTIONS:
        if a < len(pts) and b < len(pts):
            draw.line([pts[a], pts[b]], fill=(255, 255, 255, 255), width=4)

    # joints
    for x, y in pts:
        r = 3
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, 255))

    return canvas


def fallback_cutout(img_rgba):
    # keep black canvas, place the cutout on top
    canvas = Image.new("RGBA", img_rgba.size, (0, 0, 0, 255))
    canvas.alpha_composite(img_rgba)
    return canvas


files = sorted([
    p for p in INPUT_DIR.iterdir()
    if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
])

if not files:
    raise FileNotFoundError(f"No images found in {INPUT_DIR.resolve()}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Missing model file: {MODEL_PATH.resolve()}\n"
        "Download the Pose Landmarker task model and place it there."
    )

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.3,
    min_pose_presence_confidence=0.3,
)

with PoseLandmarker.create_from_options(options) as landmarker:
    for i, path in enumerate(files, start=1):
        try:
            # Load input for MediaPipe
            mp_image = mp.Image.create_from_file(str(path))
            result = landmarker.detect(mp_image)

            # For fallback rendering if no pose is detected
            img = Image.open(path).convert("RGBA")

            if result.pose_landmarks:
                out = draw_pose_on_black(result.pose_landmarks[0], img.size)
                status = "POSE"
            else:
                out = fallback_cutout(img)
                status = "FALLBACK"

            out_path = OUTPUT_DIR / f"{path.stem}.png"
            out.save(out_path)

            print(f"[{i}/{len(files)}] {status}: {path.name} -> {out_path.name}")

        except Exception as e:
            print(f"[{i}/{len(files)}] ERROR: {path.name} -> {e}")

print(f"\nDone. Saved outputs to {OUTPUT_DIR.resolve()}")
