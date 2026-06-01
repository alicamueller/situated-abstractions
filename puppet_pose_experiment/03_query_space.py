import json
from pathlib import Path
from numpy.linalg import norm

import clip
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

EMB_PATH = Path("embeddings/embeddings.npy")
FILE_PATH = Path("embeddings/filenames.json")
IMAGE_DIR = Path("images_nobg")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

image_embeddings = np.load(EMB_PATH)

with open(FILE_PATH, "r", encoding="utf-8") as f:
    filenames = json.load(f)

queries = [
    "dance",
    "dancer",
    "costume",
    "mask",
    "gesture",
    "fear",
    "ballet",
    "ritual"
]

def cosine_sim(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-9))

for query in queries:
    tokens = clip.tokenize([query]).to(device)

    with torch.no_grad():
        q_emb = model.encode_text(tokens).cpu().numpy().flatten()

    sims = np.array([cosine_sim(q_emb, img_emb.flatten()) for img_emb in image_embeddings])

    top_idx = sims.argsort()[::-1][:5]

    print(f"\nQUERY: {query}")
    for rank, idx in enumerate(top_idx, start=1):
        print(f"{rank}. {filenames[idx]}  score={sims[idx]:.3f}")

    # small result grid
    fig, axes = plt.subplots(1, 5, figsize=(15, 4), facecolor="black")
    fig.suptitle(f'CLIP Query: "{query}"', color="white")

    for ax in axes:
        ax.set_facecolor("black")
        ax.axis("off")

    for i, idx in enumerate(top_idx):
        img_path = IMAGE_DIR / filenames[idx]
        try:
            img = Image.open(img_path).convert("RGBA")
            axes[i].imshow(img)
            axes[i].set_title(f"{sims[idx]:.3f}", color="white", fontsize=8)
        except Exception:
            axes[i].text(0.5, 0.5, "No image", color="white", ha="center", va="center")

    plt.tight_layout()
    out_path = OUT_DIR / f"query_{query.replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="black")
    plt.show()

    print(f"Saved: {out_path}")
    