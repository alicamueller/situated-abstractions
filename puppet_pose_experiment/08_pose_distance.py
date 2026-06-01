from pathlib import Path
import json

import numpy as np
import pandas as pd
from numpy.linalg import norm

ORIG_EMB = Path("embeddings/embeddings.npy")
POSE_EMB = Path("embeddings/pose_embeddings.npy")

FILES = Path("embeddings/filenames.json")

orig = np.load(ORIG_EMB)
pose = np.load(POSE_EMB)

with open(FILES, "r", encoding="utf-8") as f:
    filenames = json.load(f)

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-9))

results = []

for i in range(len(orig)):

    sim = cosine_similarity(
        orig[i].flatten(),
        pose[i].flatten()
    )

    results.append({
        "filename": filenames[i],
        "similarity": sim
    })

df = pd.DataFrame(results)

# highest similarity first
df = df.sort_values(
    by="similarity",
    ascending=False
)

print("\nMOST SIMILAR\n")
print(df.head(10))

print("\nLEAST SIMILAR\n")
print(df.tail(10))

out_path = "outputs/pose_similarity.csv"
df.to_csv(out_path, index=False)

print(f"\nSaved: {out_path}")
