from pathlib import Path
import json

import umap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

ORIG_EMB_PATH = Path("embeddings/embeddings.npy")
POSE_EMB_PATH = Path("embeddings/pose_embeddings.npy")

ORIG_FILES_PATH = Path("embeddings/filenames.json")
POSE_FILES_PATH = Path("embeddings/pose_filenames.json")

ORIG_DIR = Path("images_nobg")
POSE_DIR = Path("pose_renders")

OUT_PATH = Path("outputs/combined_umap_nooverlap.png")
OUT_PATH.parent.mkdir(exist_ok=True)

# -----------------------
# Load embeddings
# -----------------------
orig_emb = np.load(ORIG_EMB_PATH)
pose_emb = np.load(POSE_EMB_PATH)

with open(ORIG_FILES_PATH, "r", encoding="utf-8") as f:
    orig_files = json.load(f)

with open(POSE_FILES_PATH, "r", encoding="utf-8") as f:
    pose_files = json.load(f)

# -----------------------
# Combine embeddings
# -----------------------
all_emb = np.vstack([orig_emb, pose_emb])

labels = (
    ["original"] * len(orig_emb)
    + ["pose"] * len(pose_emb)
)

names = orig_files + pose_files

dirs = (
    [ORIG_DIR] * len(orig_emb)
    + [POSE_DIR] * len(pose_emb)
)

# -----------------------
# UMAP
# -----------------------
reducer = umap.UMAP(
    n_components=2,
    random_state=42,
    n_neighbors=4,
    min_dist=0.5
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
# Collision avoidance
# -----------------------
min_dist = 0.14
iterations = 1500

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

    # slight pull back inward
    pts = pts * 0.997 + 0.0015

    if not moved:
        break

# Re-normalize again
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
    figsize=(12, 8),
    facecolor="black"
)

ax.set_facecolor("black")

# -----------------------
# Connection lines
# -----------------------
n = len(orig_emb)

for i in range(n):

    x1, y1 = pts[i]
    x2, y2 = pts[i + n]

    ax.plot(
        [x1, x2],
        [y1, y2],
        color="white",
        alpha=0.12,
        linewidth=1
    )

# -----------------------
# Draw thumbnails
# -----------------------
for i, (fname, img_dir, kind) in enumerate(zip(names, dirs, labels)):

    img_path = img_dir / fname

    try:
        img = Image.open(img_path).convert("RGBA")

        zoom = 0.06 if kind == "original" else 0.045

        thumb = OffsetImage(
            img,
            zoom=zoom,
            alpha=0.95 if kind == "original" else 0.80
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
            s=20,
            c="white"
        )

        print(f"Error reading {img_path.name}: {e}")

# -----------------------
# Canvas
# -----------------------
ax.set_xlim(-0.18, 1.18)
ax.set_ylim(-0.22, 1.20)

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

plt.show()

print(f"Saved to {OUT_PATH}")
