import os
import json
from pathlib import Path

import clip
import torch
import numpy as np
from PIL import Image

# ---------
# Settings
# ---------
IMAGE_DIR = Path("images_nobg")
OUT_DIR = Path("embeddings")
OUT_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# ---------
# Collect images
# ---------
image_paths = sorted([
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
])

if not image_paths:
    raise FileNotFoundError(f"No images found in {IMAGE_DIR.resolve()}")

embeddings = []
filenames = []

# ---------
# Encode images
# ---------
for i, img_path in enumerate(image_paths, start=1):
    try:
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(image)
            emb = emb.cpu().numpy().astype(np.float32)
        embeddings.append(emb)
        filenames.append(img_path.name)
        print(f"[{i}/{len(image_paths)}] OK: {img_path.name}")
    except Exception as e:
        print(f"[{i}/{len(image_paths)}] ERROR: {img_path.name} -> {e}")
        # keep shape consistent
        embeddings.append(np.zeros((1, 512), dtype=np.float32))
        filenames.append(img_path.name)

# ---------
# Save
# ---------
embeddings = np.vstack(embeddings)
np.save(OUT_DIR / "embeddings.npy", embeddings)

with open(OUT_DIR / "filenames.json", "w", encoding="utf-8") as f:
    json.dump(filenames, f, ensure_ascii=False, indent=2)

print("\nDone.")
print("Saved:")
print(f"- {OUT_DIR / 'embeddings.npy'}")
print(f"- {OUT_DIR / 'filenames.json'}")
print(f"Shape: {embeddings.shape}")
