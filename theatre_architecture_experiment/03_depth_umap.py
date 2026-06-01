from pathlib import Path
import json

import umap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

EMB_PATH = Path("embeddings/depth_embeddings.npy")
FILE_PATH = Path("embeddings/depth_filenames.json")
IMAGE_DIR = Path("depth_maps")
OUT_PATH = Path("outputs/depth_umap_overlap.png")
OUT_PATH.parent.mkdir(exist_ok=True)

embeddings = np.load(EMB_PATH)

with open(FILE_PATH, "r", encoding="utf-8") as f:
    filenames = json.load(f)

reducer = umap.UMAP(
    n_components=2,
    random_state=42,
    n_neighbors=4,
    min_dist=0.3
)

coords = reducer.fit_transform(embeddings)

pts = coords.astype(float)
pts[:, 0] = (pts[:, 0] - pts[:, 0].min()) / (pts[:, 0].max() - pts[:, 0].min() + 1e-9)
pts[:, 1] = (pts[:, 1] - pts[:, 1].min()) / (pts[:, 1].max() - pts[:, 1].min() + 1e-9)

fig, ax = plt.subplots(figsize=(10, 7), facecolor="black")
ax.set_facecolor("black")

for i, fname in enumerate(filenames):
    img_path = IMAGE_DIR / fname

    try:
        img = Image.open(img_path).convert("L")

        # only mild upscaling to reduce pixelation
        img = img.resize(
            (img.width * 2, img.height * 2),
            Image.Resampling.LANCZOS
        )

        thumb = OffsetImage(
            np.array(img),
            zoom=0.075,
            cmap="gray",
            alpha=0.9
        )

        ab = AnnotationBbox(
            thumb,
            (pts[i, 0], pts[i, 1]),
            frameon=False,
            pad=0.0
        )

        ax.add_artist(ab)

    except Exception as e:
        ax.scatter(pts[i, 0], pts[i, 1], s=16, c="white")
        print(f"Error reading {img_path.name}: {e}")

ax.set_xlim(-0.10, 1.10)
ax.set_ylim(-0.10, 1.10)
ax.axis("off")

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

plt.savefig(
    OUT_PATH,
    dpi=150,
    bbox_inches="tight",
    facecolor="black"
)

plt.close()
print(f"Saved to {OUT_PATH}")
