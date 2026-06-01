from pathlib import Path

import cv2

INPUT_DIR = Path("images")
OUTPUT_DIR = Path("edges")
OUTPUT_DIR.mkdir(exist_ok=True)

image_paths = sorted([
    p for p in INPUT_DIR.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
])

if not image_paths:
    raise FileNotFoundError(f"No images found in {INPUT_DIR.resolve()}")

for i, img_path in enumerate(image_paths, start=1):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError("cv2 could not read image")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(
            gray,
            threshold1=80,
            threshold2=180
        )

        out_path = OUTPUT_DIR / f"{img_path.stem}.png"
        cv2.imwrite(str(out_path), edges)

        print(f"[{i}/{len(image_paths)}] OK: {img_path.name} -> {out_path.name}")

    except Exception as e:
        print(f"[{i}/{len(image_paths)}] ERROR: {img_path.name} -> {e}")

print(f"\nDone. Saved edge maps to {OUTPUT_DIR.resolve()}")
