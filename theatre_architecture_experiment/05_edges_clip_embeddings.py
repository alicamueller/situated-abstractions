from pathlib import Path
import json

import clip
import torch
import numpy as np
from PIL import Image

IMAGE_DIR = Path("edges")
OUT_DIR = Path("embeddings")
OUT_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

image_paths = sorted([
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
])

if not image_paths:
    raise FileNotFoundError(f"No images found in {IMAGE_DIR.resolve()}")

embeddings = []
filenames = []

for i, img_path in enumerate(image_paths, start=1):
    try:
        img = Image.open(img_path).convert("RGB")
        image = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model.encode_image(image).cpu().numpy().astype(np.float32)

        embeddings.append(emb)
        filenames.append(img_path.name)
        print(f"[{i}/{len(image_paths)}] OK: {img_path.name}")

    except Exception as e:
        print(f"[{i}/{len(image_paths)}] ERROR: {img_path.name} -> {e}")
        embeddings.append(np.zeros((1, 512), dtype=np.float32))
        filenames.append(img_path.name)

embeddings = np.vstack(embeddings)
np.save(OUT_DIR / "edges_embeddings.npy", embeddings)

with open(OUT_DIR / "edges_filenames.json", "w", encoding="utf-8") as f:
    json.dump(filenames, f, ensure_ascii=False, indent=2)

print("\nDone.")
print(f"Saved: {OUT_DIR / 'edges_embeddings.npy'}")
print(f"Saved: {OUT_DIR / 'edges_filenames.json'}")
print(f"Shape: {embeddings.shape}")
