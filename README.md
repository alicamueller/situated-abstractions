# Situated Abstractions

**Situated Abstractions** is a series of experimental studies investigating how machine vision systems transform culturally situated visual artefacts and architectural representations into machine-readable abstractions.
## Research Question

**How do machine vision systems reorganize cultural artefacts under processes of visual abstraction, and what forms of meaning remain legible after this transformation?**

## Project Structure

### 1. Artefact Pose Experiment

A collection of cultural artefacts, puppets, costumes, figurines, figural drawings is transformed through human pose estimation using MediaPipe Pose. 

Pipeline:

```text
Original Artefact
      ↓
MediaPipe Pose Estimation
      ↓
Pose Skeleton
      ↓
CLIP Embedding
      ↓
UMAP Projection
```

The experiment compares CLIP embeddings of original artefacts with their pose-reduced counterparts.
By measuring semantic displacement between original and pose-derived embeddings, the study investigates which aspects of cultural artefacts remain legible when visual surface, ornament, and materiality are reduced to standardized body geometries.
### 2. Theatre Space Experiment

A corpus of theatre architectural drawings and paintings is transformed through two forms of spatial abstraction:

- Monocular Depth Estimation
- Edge Detection
    
Pipeline:
```text
Original Interior
      ↓
Depth Estimation / Edge Detection
      ↓
Spatial Abstraction
      ↓
CLIP Embedding
      ↓
UMAP Projection
```

The resulting embedding spaces are compared to examine how different abstraction techniques reorganize architectural representations.
The experiment investigates which spatial features dominate machine vision when texture, ornament, and material detail are progressively removed.
## Methods

### Visual Embeddings
All images are embedded using OpenAI CLIP (ViT-B/32).
### Dimensionality Reduction
Embedding spaces are projected into two dimensions using UMAP.
### Abstraction Layers
The project compares multiple forms of computational abstraction:

|Domain|Original Representation|Abstraction|
|---|---|---|
|Cultural Artefacts|Object Image|Pose Skeleton|
|Theatre Architecture|Interior Image|Depth Map|
|Theatre Architecture|Interior Image|Edge Map|
### Similarity Analysis
Semantic displacement is measured through cosine similarity between original and abstracted representations.
## Key Idea
The project treats pose estimation, depth estimation, and edge detection not merely as technical operations, but as **reduction interfaces**: computational procedures that transform culturally complex visual objects into standardized relational structures.

The resulting embedding spaces reveal how machine vision systems privilege certain forms of visual information while suppressing others.

## Technologies
- Python
- CLIP (ViT-B/32)
- PyTorch
- UMAP
- MediaPipe Pose
- Transformers
- Depth Anything V2 / MiDaS
- OpenCV
- NumPy
- PIL
- Matplotlib

## Outputs
The repository contains:
- CLIP embedding pipelines
- UMAP visualizations
- Pose abstraction experiments
- Depth abstraction experiments
- Edge abstraction experiments
- Comparative embedding analyses
## Data
The image datasets used in these experiments are not included in this repository.

Materials were obtained from:
- Victoria and Albert Museum Online Collection
- Ashmolean Museum Online Collection

