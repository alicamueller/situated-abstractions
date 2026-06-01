from pathlib import Path
import json

import umap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

DEPTH_EMB_PATH = Path("embeddings/depth_embeddings.npy")
EDGE_EMB_PATH = Path("embeddings/edges_embeddings.npy")

DEPTH_FILES_PATH = Path("embeddings/depth_filenames.json")
EDGE_FILES_PATH = Path("embeddings/edges_filenames.json")

DEPTH_DIR = Path("depth_maps")
EDGE_DIR = Path("edges")

OUT_PATH = Path("outputs/combined_depth_edges.png")
OUT_PATH.parent.mkdir(exist_ok=True)

# -----------------------
# Load embeddings
# -----------------------
depth_emb = np.load(DEPTH_EMB_PATH)
edge_emb = np.load(EDGE_EMB_PATH)

with open(DEPTH_FILES_PATH, "r", encoding="utf-8") as f:
    depth_files = json.load(f)

with open(EDGE_FILES_PATH, "r", encoding="utf-8") as f:
    edge_files = json.load(f)

if len(depth_emb) != len(edge_emb):
    raise ValueError("Depth and edge embedding counts do not match.")

# Optional sanity check: same filenames in same order
if depth_files != edge_files:
    print("Warning: filenames are not identical in order.")
    print("The plot will still run, but check the pairing carefully.")

# -----------------------
# Combine
# -----------------------
all_emb = np.vstack([depth_emb, edge_emb])

labels = ["depth"] * len(depth_emb) + ["edge"] * len(edge_emb)
names = depth_files + edge_files
dirs = [DEPTH_DIR] * len(depth_emb) + [EDGE_DIR] * len(edge_emb)

# -----------------------
# UMAP
# -----------------------
reducer = umap.UMAP(
    n_components=2,
    random_state=42,
    n_neighbors=4,
    min_dist=0.4
)

coords = reducer.fit_transform(all_emb)

# Normalize to 0..1
pts = coords.astype(float)
pts[:, 0] = (pts[:, 0] - pts[:, 0].min()) / (pts[:, 0].max() - pts[:, 0].min() + 1e-9)
pts[:, 1] = (pts[:, 1] - pts[:, 1].min()) / (pts[:, 1].max() - pts[:, 1].min() + 1e-9)

# -----------------------
# Plot
# -----------------------
fig, ax = plt.subplots(figsize=(12, 8), facecolor="black")
ax.set_facecolor("black")

# Lines connecting depth and edge versions of the same item
n = len(depth_emb)
for i in range(n):
    x1, y1 = pts[i]
    x2, y2 = pts[i + n]
    ax.plot([x1, x2], [y1, y2], color="white", alpha=0.12, linewidth=1)

# Thumbnails
for i, (fname, img_dir, kind) in enumerate(zip(names, dirs, labels)):
    img_path = img_dir / fname
    try:
        img = Image.open(img_path).convert("L")

        # mild upscaling to reduce pixelation
        img = img.resize(
            (img.width * 2, img.height * 2),
            Image.Resampling.LANCZOS
        )

        zoom = 0.075 if kind == "depth" else 0.07

        thumb = OffsetImage(
            np.array(img),
            zoom=zoom,
            cmap="gray",
            alpha=0.9 if kind == "depth" else 0.85
        )

        ab = AnnotationBbox(
            thumb,
            (pts[i, 0], pts[i, 1]),
            frameon=False,
            pad=0.0
        )
        ax.add_artist(ab)

    except Exception as e:
        ax.scatter(
            pts[i, 0],
            pts[i, 1],
            s=16,
            c="white" if kind == "edge" else "cyan"
        )
        print(f"Error reading {img_path.name}: {e}")

# Legend text
ax.text(0.02, 0.98, "depth", color="cyan", transform=ax.transAxes, va="top", fontsize=10)
ax.text(0.02, 0.95, "edge", color="white", transform=ax.transAxes, va="top", fontsize=10)

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
