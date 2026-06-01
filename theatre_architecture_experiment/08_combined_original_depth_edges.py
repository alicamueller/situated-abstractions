from pathlib import Path
import json

import umap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# -----------------------
# Paths
# -----------------------
ORIG_EMB_PATH = Path("embeddings/original_embeddings.npy")
DEPTH_EMB_PATH = Path("embeddings/depth_embeddings.npy")
EDGE_EMB_PATH = Path("embeddings/edges_embeddings.npy")

ORIG_FILES_PATH = Path("embeddings/original_filenames.json")
DEPTH_FILES_PATH = Path("embeddings/depth_filenames.json")
EDGE_FILES_PATH = Path("embeddings/edges_filenames.json")

ORIG_DIR = Path("images")
DEPTH_DIR = Path("depth_maps")
EDGE_DIR = Path("edges")

OUT_PATH = Path("outputs/combined_original_depth_edges.png")
OUT_PATH.parent.mkdir(exist_ok=True)

# -----------------------
# Load embeddings
# -----------------------
orig_emb = np.load(ORIG_EMB_PATH)
depth_emb = np.load(DEPTH_EMB_PATH)
edge_emb = np.load(EDGE_EMB_PATH)

with open(ORIG_FILES_PATH, "r", encoding="utf-8") as f:
    orig_files = json.load(f)

with open(DEPTH_FILES_PATH, "r", encoding="utf-8") as f:
    depth_files = json.load(f)

with open(EDGE_FILES_PATH, "r", encoding="utf-8") as f:
    edge_files = json.load(f)

n = len(orig_emb)

# -----------------------
# Combine all embeddings
# -----------------------
all_emb = np.vstack([
    orig_emb,
    depth_emb,
    edge_emb
])

labels = (
    ["original"] * n +
    ["depth"] * n +
    ["edge"] * n
)

names = orig_files + depth_files + edge_files

dirs = (
    [ORIG_DIR] * n +
    [DEPTH_DIR] * n +
    [EDGE_DIR] * n
)

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

# Normalize
pts = coords.astype(float)

pts[:, 0] = (
    (pts[:, 0] - pts[:, 0].min()) /
    (pts[:, 0].max() - pts[:, 0].min() + 1e-9)
)

pts[:, 1] = (
    (pts[:, 1] - pts[:, 1].min()) /
    (pts[:, 1].max() - pts[:, 1].min() + 1e-9)
)

# -----------------------
# Plot
# -----------------------
fig, ax = plt.subplots(
    figsize=(13, 9),
    facecolor="black"
)

ax.set_facecolor("black")

# -----------------------
# Connection lines
# -----------------------
for i in range(n):

    # original -> depth
    x1, y1 = pts[i]
    x2, y2 = pts[i + n]

    ax.plot(
        [x1, x2],
        [y1, y2],
        color="cyan",
        alpha=0.10,
        linewidth=1
    )

    # original -> edge
    x3, y3 = pts[i + 2 * n]

    ax.plot(
        [x1, x3],
        [y1, y3],
        color="white",
        alpha=0.08,
        linewidth=1
    )

# -----------------------
# Draw images
# -----------------------
for i, (fname, img_dir, kind) in enumerate(zip(names, dirs, labels)):

    img_path = img_dir / fname

    try:

        if kind == "original":
            img = Image.open(img_path).convert("RGB")

        else:
            img = Image.open(img_path).convert("L")

            img = img.resize(
                (img.width * 2, img.height * 2),
                Image.Resampling.LANCZOS
            )

        # different sizes
        if kind == "original":
            zoom = 0.08
            alpha = 0.95

        elif kind == "depth":
            zoom = 0.07
            alpha = 0.85

        else:
            zoom = 0.065
            alpha = 0.75

        thumb = OffsetImage(
            np.array(img),
            zoom=zoom,
            cmap="gray" if kind != "original" else None,
            alpha=alpha
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
            c="white"
        )

        print(f"Error reading {img_path.name}: {e}")

# -----------------------
# Labels
# -----------------------
ax.text(
    0.02,
    0.98,
    "original",
    color="orange",
    transform=ax.transAxes,
    va="top",
    fontsize=10
)

ax.text(
    0.02,
    0.95,
    "depth",
    color="cyan",
    transform=ax.transAxes,
    va="top",
    fontsize=10
)

ax.text(
    0.02,
    0.92,
    "edge",
    color="white",
    transform=ax.transAxes,
    va="top",
    fontsize=10
)

# -----------------------
# Canvas
# -----------------------
ax.set_xlim(-0.10, 1.10)
ax.set_ylim(-0.10, 1.10)

ax.axis("off")

plt.subplots_adjust(
    left=0,
    right=1,
    top=1,
    bottom=0
)

plt.savefig(
    OUT_PATH,
    dpi=150,
    bbox_inches="tight",
    facecolor="black"
)

plt.close()

print(f"Saved to {OUT_PATH}")
