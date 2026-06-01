import json
from pathlib import Path

import umap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

EMB_PATH = Path("embeddings/embeddings.npy")
FILE_PATH = Path("embeddings/filenames.json")
IMAGE_DIR = Path("images_nobg")
OUT_PATH = Path("outputs/umap_20_nooverlap.png")
OUT_PATH.parent.mkdir(exist_ok=True)

embeddings = np.load(EMB_PATH)

with open(FILE_PATH, "r", encoding="utf-8") as f:
    filenames = json.load(f)

reducer = umap.UMAP(
    n_components=2,
    random_state=42,
    n_neighbors=3,
    min_dist=0.8
)
coords = reducer.fit_transform(embeddings)

# Normalize to 0..1
pts = coords.astype(float)
pts[:, 0] = (pts[:, 0] - pts[:, 0].min()) / (pts[:, 0].max() - pts[:, 0].min() + 1e-9)
pts[:, 1] = (pts[:, 1] - pts[:, 1].min()) / (pts[:, 1].max() - pts[:, 1].min() + 1e-9)

# Push overlapping points apart
min_dist = 0.30
iterations = 1000

for _ in range(iterations):
    moved = False

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx = pts[j, 0] - pts[i, 0]
            dy = pts[j, 1] - pts[i, 1]
            dist = (dx * dx + dy * dy) ** 0.5 + 1e-9

            if dist < min_dist:
                overlap = min_dist - dist
                ux = dx / dist
                uy = dy / dist
                push = overlap * 0.5

                pts[i, 0] -= ux * push
                pts[i, 1] -= uy * push
                pts[j, 0] += ux * push
                pts[j, 1] += uy * push

                moved = True

    # keep cloud compact
    pts = pts * 0.995 + 0.0025

    if not moved:
        break

# Re-normalize so all points stay visible
pts[:, 0] = (pts[:, 0] - pts[:, 0].min()) / (pts[:, 0].max() - pts[:, 0].min() + 1e-9)
pts[:, 1] = (pts[:, 1] - pts[:, 1].min()) / (pts[:, 1].max() - pts[:, 1].min() + 1e-9)

fig, ax = plt.subplots(figsize=(10, 7), facecolor="black")
ax.set_facecolor("black")

for i, fname in enumerate(filenames):
    img_path = IMAGE_DIR / fname
    try:
        img = Image.open(img_path).convert("RGBA")
        thumb = OffsetImage(img, zoom=0.065)
        ab = AnnotationBbox(
            thumb,
            (pts[i, 0], pts[i, 1]),
            frameon=False,
            pad=0.0
        )
        ax.add_artist(ab)
    except Exception:
        ax.scatter(pts[i, 0], pts[i, 1], s=20, c="white")

# More breathing room so nothing gets cut off
ax.set_xlim(-0.18, 1.18)
ax.set_ylim(-0.22, 1.20)

ax.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="black")
plt.show()

print(f"Saved to {OUT_PATH}")
