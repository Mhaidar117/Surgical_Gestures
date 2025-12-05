# Embedding Extraction and Visualization Guide

This guide shows you how to extract and visualize embeddings from your trained ViT model.

## Overview

The pipeline has two steps:
1. **Extract embeddings** from a trained model checkpoint
2. **Visualize embeddings** using dimensionality reduction (t-SNE, UMAP, PCA)

## Available Embeddings

The extraction script extracts embeddings at multiple levels:

| Embedding Type | Shape | Description |
|---------------|-------|-------------|
| `vit_early` | (B, T, 384) | Early ViT layer activations (block 3/12) |
| `vit_mid` | (B, T, 384) | Mid ViT layer activations (block 6/12) |
| `vit_late` | (B, T, 384) | Late ViT layer activations (block 11/12) |
| `cls_tokens` | (B, T, 384) | CLS tokens after ViT (frame-level) |
| `temporal_memory` | (B, T, 384) | After temporal transformer (context-aware) |
| `mean_pooled` | (B, 384) | Mean-pooled trial-level representation |
| `attn_pooled` | (B, 384) | Attention-pooled trial-level representation |
| `projected` | (B, T, 384) | Projected latent space (pre-decoder) |

Where:
- `B` = batch size (number of gesture segments)
- `T` = temporal window (number of frames, typically 10-30)
- `384` = embedding dimension

## Step 1: Extract Embeddings

### Basic Usage

```bash
python src/inference/extract_embeddings.py \
  --checkpoint checkpoints/checkpoint_epoch_80.pth \
  --config configs/your_config.yaml \
  --data_root /path/to/JIGSAWS \
  --task Knot_Tying \
  --output embeddings/knot_tying_embeddings.pkl
```

### Advanced Options

```bash
python src/inference/extract_embeddings.py \
  --checkpoint checkpoints/checkpoint_epoch_80.pth \
  --config configs/your_config.yaml \
  --data_root /path/to/JIGSAWS \
  --task Knot_Tying \
  --output embeddings/knot_tying_embeddings.pkl \
  --batch_size 8 \
  --max_batches 50 \
  --layers early mid late
```

**Arguments:**
- `--checkpoint`: Path to trained model checkpoint (`.pth` file)
- `--config`: Path to training config YAML file
- `--data_root`: Root directory containing `Gestures/` folder
- `--task`: Task name (`Knot_Tying`, `Needle_Passing`, or `Suturing`)
- `--output`: Where to save extracted embeddings (`.pkl` file)
- `--batch_size`: Batch size for extraction (default: 8)
- `--max_batches`: Limit number of batches to process (default: all)
- `--layers`: Which ViT layers to extract (default: `early mid late`)

## Step 2: Visualize Embeddings

### Visualize Single Embedding Type

```bash
# 2D t-SNE of mean-pooled embeddings, colored by gesture
python src/inference/visualize_embeddings.py \
  --embeddings embeddings/knot_tying_embeddings.pkl \
  --embedding_type mean_pooled \
  --label_type gesture \
  --method tsne \
  --output_dir plots/
```

### Compare All Embedding Types

```bash
# Compare all embeddings in a single figure
python src/inference/visualize_embeddings.py \
  --embeddings embeddings/knot_tying_embeddings.pkl \
  --label_type gesture \
  --method tsne \
  --output_dir plots/ \
  --compare_all
```

### 3D Visualization

```bash
# 3D UMAP visualization
python src/inference/visualize_embeddings.py \
  --embeddings embeddings/knot_tying_embeddings.pkl \
  --embedding_type temporal_memory \
  --label_type skill \
  --method umap \
  --n_components 3 \
  --output_dir plots/
```

**Arguments:**
- `--embeddings`: Path to embeddings pickle file
- `--embedding_type`: Which embedding to visualize (see table above)
- `--label_type`: Color by `gesture` or `skill` labels
- `--method`: Dimensionality reduction method (`tsne`, `umap`, or `pca`)
- `--n_components`: Number of dimensions (2 or 3)
- `--output_dir`: Directory to save plots
- `--aggregate`: How to aggregate temporal dimension (`mean`, `max`, `first`, `last`)
- `--compare_all`: Compare all embedding types in a grid

## Example Workflow

### Full Pipeline

```bash
# 1. Extract embeddings from trained model
python src/inference/extract_embeddings.py \
  --checkpoint checkpoints/checkpoint_epoch_80.pth \
  --config configs/base_config.yaml \
  --data_root /path/to/JIGSAWS \
  --task Knot_Tying \
  --output embeddings/knot_tying.pkl \
  --batch_size 16

# 2. Visualize all embeddings colored by gesture
python src/inference/visualize_embeddings.py \
  --embeddings embeddings/knot_tying.pkl \
  --label_type gesture \
  --method tsne \
  --compare_all \
  --output_dir plots/gesture_analysis/

# 3. Visualize specific embedding colored by skill
python src/inference/visualize_embeddings.py \
  --embeddings embeddings/knot_tying.pkl \
  --embedding_type mean_pooled \
  --label_type skill \
  --method umap \
  --output_dir plots/skill_analysis/

# 4. 3D visualization of temporal memory
python src/inference/visualize_embeddings.py \
  --embeddings embeddings/knot_tying.pkl \
  --embedding_type temporal_memory \
  --label_type gesture \
  --method umap \
  --n_components 3 \
  --output_dir plots/temporal_3d/
```

## Interpretation Guide

### What to Look For

1. **Clustering by Gesture**: Good separation between gestures indicates the model learned distinct representations
2. **Skill Separation**: If skilled vs novice form separate clusters, the model captured expertise differences
3. **Layer Progression**: Compare `vit_early`, `vit_mid`, `vit_late` to see how representations evolve
4. **Temporal vs Pooled**: Compare `cls_tokens` (frame-level) vs `temporal_memory` (context-aware) vs `mean_pooled` (trial-level)

### Recommended Visualizations

| Research Question | Embedding Type | Label Type | Method |
|------------------|----------------|------------|--------|
| "Does my model distinguish gestures?" | `mean_pooled` | gesture | t-SNE |
| "Which layer captures gesture info?" | Compare `vit_early/mid/late` | gesture | t-SNE |
| "Does temporal context help?" | Compare `cls_tokens` vs `temporal_memory` | gesture | t-SNE |
| "Can we predict skill level?" | `attn_pooled` | skill | UMAP |
| "What does the decoder see?" | `projected` | gesture | PCA |

## Using Embeddings in Python

You can also load and use embeddings programmatically:

```python
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load embeddings
with open('embeddings/knot_tying.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract what you need
mean_pooled = data['mean_pooled']  # (N, 384)
gesture_labels = data['gesture_label']  # (N,)
trial_ids = data['trial_id']  # List of N trial IDs

print(f"Mean pooled shape: {mean_pooled.shape}")
print(f"Number of samples: {len(gesture_labels)}")
print(f"Unique gestures: {np.unique(gesture_labels)}")

# Reduce dimensionality
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d = tsne.fit_transform(mean_pooled)

# Plot
plt.figure(figsize=(10, 8))
for gesture in np.unique(gesture_labels):
    mask = gesture_labels == gesture
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                label=f'G{gesture+1}', alpha=0.7)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('Gesture Embeddings')
plt.legend()
plt.show()
```

## Dependencies

Make sure you have these packages installed:

```bash
pip install numpy matplotlib scikit-learn seaborn
pip install umap-learn  # For UMAP (optional but recommended)
```

## Troubleshooting

### Out of Memory
If you run out of memory during extraction:
- Reduce `--batch_size` (e.g., to 4 or 2)
- Use `--max_batches` to process only a subset
- Extract fewer layers with `--layers mid` instead of `early mid late`

### UMAP Not Available
If you get a UMAP error:
```bash
pip install umap-learn
```

### Empty Plots
If plots are empty or have errors:
- Check that the embedding file contains data: `--embeddings path/to/file.pkl`
- Verify the embedding type exists: look at the "Available embeddings" printout
- Try a different aggregation method: `--aggregate mean` or `--aggregate first`

## Tips for Best Results

1. **Use more data**: Extract from entire dataset (remove `--max_batches` limit)
2. **Try different methods**: t-SNE good for clusters, UMAP preserves global structure, PCA for linear relationships
3. **Adjust perplexity**: For t-SNE, use perplexity ~5-50 depending on dataset size
4. **Color by different labels**: Compare gesture vs skill separation
5. **Save high-res**: Plots saved at 300 DPI for publication quality
