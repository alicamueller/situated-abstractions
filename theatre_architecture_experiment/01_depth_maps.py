from pathlib import Path

import numpy as np
from PIL import Image
from transformers import pipeline

INPUT_DIR = Path("images")
OUTPUT_DIR = Path("depth_maps")
OUTPUT_DIR.mkdir(exist_ok=True)

# Relative depth model, zero-shot, good for spatial abstraction
checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
pipe = pipeline("depth-estimation", model=checkpoint)

image_paths = sorted([
    p for p in INPUT_DIR.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
])

if not image_paths:
    raise FileNotFoundError(f"No images found in {INPUT_DIR.resolve()}")

for i, img_path in enumerate(image_paths, start=1):
    try:
        image = Image.open(img_path).convert("RGB")
        result = pipe(image)

        # Hugging Face docs say the pipeline returns:
        # - predicted_depth: tensor
        # - depth: visualized PIL image
        depth = result["predicted_depth"].squeeze().cpu().numpy()

        # Normalize to 0..255 for saving
        dmin, dmax = float(depth.min()), float(depth.max())
        depth_norm = (depth - dmin) / (dmax - dmin + 1e-9)
        depth_img = Image.fromarray((depth_norm * 255).astype(np.uint8), mode="L")

        out_path = OUTPUT_DIR / f"{img_path.stem}.png"
        depth_img.save(out_path)

        print(f"[{i}/{len(image_paths)}] OK: {img_path.name} -> {out_path.name}")

    except Exception as e:
        print(f"[{i}/{len(image_paths)}] ERROR: {img_path.name} -> {e}")

print(f"\nDone. Saved depth maps to {OUTPUT_DIR.resolve()}")
