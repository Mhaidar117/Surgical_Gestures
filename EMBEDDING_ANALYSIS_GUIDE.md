# Complete Guide: Viewing Gesture Embeddings from Trained Model

## Quick Start (5 minutes)

If you just want to see embeddings quickly:

```bash
# Edit paths in examples/quick_embedding_analysis.py, then run:
python examples/quick_embedding_analysis.py
```

This will generate a 4-panel visualization showing t-SNE and PCA plots colored by gesture and skill.

---

## Complete Workflow

### Step 1: Extract Embeddings from Trained Model

```bash
python src/inference/extract_embeddings.py \
  --checkpoint checkpoints/baseline_psm2/checkpoint_epoch_50.pth \
  --config src/configs/baseline.yaml \
  --data_root  ./\
  --task Knot_Tying \
  --output embeddings/knot_tying_embeddings.pkl \
  --batch_size 16
```

**What this extracts:**
- `vit_early/mid/late`: ViT layer activations at different depths
- `cls_tokens`: Frame-level embeddings (B, T, 384)
- `temporal_memory`: Context-aware embeddings after temporal transformer
- `mean_pooled`: Trial-level representation (B, 384) ← **Most useful**
- `attn_pooled`: Attention-pooled trial representation
- `projected`: Pre-decoder latent space

### Step 2: Visualize Embeddings

#### Option A: Compare All Embeddings (Recommended First Step)

```bash
python src/inference/visualize_embeddings.py \
  --embeddings embeddings/knot_tying_embeddings.pkl \
  --label_type gesture \
  --method tsne \
  --compare_all \
  --output_dir plots/overview/
```

This creates a grid comparing all embedding types in one figure.

#### Option B: Single Embedding Type (Detailed Analysis)

```bash
# 2D t-SNE of mean-pooled embeddings
python src/inference/visualize_embeddings.py \
  --embeddings embeddings/knot_tying_embeddings.pkl \
  --embedding_type mean_pooled \
  --label_type gesture \
  --method tsne \
  --output_dir plots/

# 3D UMAP visualization
python src/inference/visualize_embeddings.py \
  --embeddings embeddings/knot_tying_embeddings.pkl \
  --embedding_type temporal_memory \
  --label_type gesture \
  --method umap \
  --n_components 3 \
  --output_dir plots/
```

#### Option C: Analyze Gesture Similarity

```bash
python examples/analyze_gesture_similarity.py \
  --embeddings embeddings/knot_tying_embeddings.pkl \
  --embedding_type mean_pooled \
  --metric cosine \
  --output_dir plots/similarity/
```

This generates:
- Similarity matrix heatmap
- Hierarchical clustering dendrogram
- List of most/least similar gesture pairs
- Within-gesture variance analysis

---

## Understanding the Outputs

### Embedding Types Explained

| Type | Shape | When to Use |
|------|-------|-------------|
| `mean_pooled` | (B, 384) | **Best for**: Overall gesture clustering, skill analysis |
| `temporal_memory` | (B, T, 384) | **Best for**: Understanding temporal dynamics |
| `vit_mid` | (B, T, 384) | **Best for**: Comparing layer representations |
| `projected` | (B, T, 384) | **Best for**: Understanding what decoder receives |

### What Good Clustering Looks Like

✅ **Good clustering**:
- Clear separation between different gestures
- Tight clusters (low within-gesture variance)
- Similar gestures closer together (e.g., G1 and G2 if they're related)

❌ **Poor clustering**:
- All gestures mixed together
- Very diffuse clusters
- No clear structure

### Dimensionality Reduction Methods

- **t-SNE**: Best for visualizing local structure and clusters
  - Good at: Separating distinct clusters
  - Bad at: Preserving global distances
  - Use when: You want to see if gestures form separate groups

- **UMAP**: Best balance of local and global structure
  - Good at: Both clusters and overall topology
  - Bad at: Can be slower than t-SNE
  - Use when: You want to see relationships between clusters

- **PCA**: Best for understanding linear structure
  - Good at: Showing variance explained, computational speed
  - Bad at: Nonlinear relationships
  - Use when: You want interpretable axes or quick exploration

---

## Example Use Cases

### Use Case 1: "Does my model learn distinct gestures?"

```bash
# Extract embeddings
python src/inference/extract_embeddings.py \
  --checkpoint checkpoints/best_model.pth \
  --config configs/config.yaml \
  --data_root /path/to/JIGSAWS \
  --task Knot_Tying \
  --output embeddings/knot_tying.pkl

# Visualize with t-SNE
python src/inference/visualize_embeddings.py \
  --embeddings embeddings/knot_tying.pkl \
  --embedding_type mean_pooled \
  --label_type gesture \
  --method tsne \
  --output_dir plots/
```

**Look for**: Clear separation between gesture clusters

### Use Case 2: "Which gestures are easily confused?"

```bash
# Analyze similarity
python examples/analyze_gesture_similarity.py \
  --embeddings embeddings/knot_tying.pkl \
  --embedding_type mean_pooled \
  --output_dir plots/similarity/
```

**Look for**: High similarity scores in the heatmap indicate confusion risk

### Use Case 3: "How do representations evolve through layers?"

```bash
# Extract all layers
python src/inference/extract_embeddings.py \
  --checkpoint checkpoints/best_model.pth \
  --config configs/config.yaml \
  --data_root /path/to/JIGSAWS \
  --task Knot_Tying \
  --layers early mid late \
  --output embeddings/all_layers.pkl

# Compare layers
python src/inference/visualize_embeddings.py \
  --embeddings embeddings/all_layers.pkl \
  --label_type gesture \
  --method tsne \
  --compare_all \
  --output_dir plots/layers/
```

**Look for**: Increasing gesture separation from early → mid → late layers

### Use Case 4: "Does temporal context help?"

```bash
# Compare frame-level vs context-aware embeddings
python src/inference/visualize_embeddings.py \
  --embeddings embeddings/knot_tying.pkl \
  --embedding_type cls_tokens \
  --aggregate first \
  --label_type gesture \
  --method tsne \
  --output_dir plots/comparison/

python src/inference/visualize_embeddings.py \
  --embeddings embeddings/knot_tying.pkl \
  --embedding_type temporal_memory \
  --aggregate mean \
  --label_type gesture \
  --method tsne \
  --output_dir plots/comparison/
```

**Look for**: Better clustering in temporal_memory vs cls_tokens

### Use Case 5: "Can we predict skill level from embeddings?"

```bash
# Visualize colored by skill
python src/inference/visualize_embeddings.py \
  --embeddings embeddings/knot_tying.pkl \
  --embedding_type attn_pooled \
  --label_type skill \
  --method umap \
  --output_dir plots/skill/
```

**Look for**: Separation between Novice, Intermediate, and Expert

---

## Using Embeddings Programmatically

If you want to use embeddings in your own analysis:

```python
import pickle
import numpy as np

# Load embeddings
with open('embeddings/knot_tying_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

# Access different embeddings
mean_pooled = data['mean_pooled']        # (N, 384) - trial-level
temporal = data['temporal_memory']       # (N, T, 384) - frame-level context-aware
gesture_labels = data['gesture_label']   # (N,)
skill_labels = data['skill_label']       # (N,)
trial_ids = data['trial_id']             # List[str]

# Example: Compute average embedding per gesture
for gesture in range(15):
    mask = gesture_labels == gesture
    if mask.any():
        centroid = mean_pooled[mask].mean(axis=0)
        print(f"G{gesture+1} centroid shape: {centroid.shape}")

# Example: Find most similar gestures to G1
from scipy.spatial.distance import cosine
g1_mask = gesture_labels == 0
g1_centroid = mean_pooled[g1_mask].mean(axis=0)

similarities = []
for gesture in range(1, 15):
    mask = gesture_labels == gesture
    if mask.any():
        centroid = mean_pooled[mask].mean(axis=0)
        sim = 1 - cosine(g1_centroid, centroid)
        similarities.append((gesture+1, sim))

similarities.sort(key=lambda x: x[1], reverse=True)
print(f"\nGestures most similar to G1:")
for g, sim in similarities[:5]:
    print(f"  G{g}: {sim:.4f}")
```

---

## Extracting from Specific Locations in Model Code

If you want to extract embeddings at a custom location, here's where to look:

### Frame-level CLS tokens
**File**: `src/models/visual.py:222`
```python
cls_tokens = features[:, 0, :]  # (B*T, D)
```

### Temporal transformer output
**File**: `src/models/temporal_transformer.py:96`
```python
memory = self.encoder(seq, src_key_padding_mask=key_padding_mask)
```

### Pre-decoder projection
**File**: `src/models/kinematics.py:213`
```python
projected = self.projection(pooled_emb)  # (B, T, D)
```

### During training forward pass
**File**: `src/training/train_vit_system.py:140-142`
```python
if return_embeddings:
    outputs['embeddings'] = emb_layers
    outputs['memory'] = memory
```

---

## Troubleshooting

### "CUDA out of memory" during extraction
```bash
# Reduce batch size
python src/inference/extract_embeddings.py \
  --batch_size 4 \
  ...

# Or extract only a subset
python src/inference/extract_embeddings.py \
  --max_batches 20 \
  ...
```

### "No module named 'umap'"
```bash
pip install umap-learn
```

### Empty or weird plots
- Check that embeddings file exists and is not corrupted
- Try different aggregation: `--aggregate mean` vs `--aggregate first`
- Try different reduction method: t-SNE vs UMAP vs PCA
- Adjust perplexity for t-SNE: smaller for fewer samples

### Too few samples in visualization
- Remove `--max_batches` limit when extracting
- Check that your dataset has enough samples

---

## File Locations

- **Extraction script**: `src/inference/extract_embeddings.py`
- **Visualization script**: `src/inference/visualize_embeddings.py`
- **Quick analysis**: `examples/quick_embedding_analysis.py`
- **Similarity analysis**: `examples/analyze_gesture_similarity.py`
- **Detailed guide**: `src/inference/README_embeddings.md`

---

## Next Steps

After viewing embeddings, you might want to:

1. **Train a classifier**: Use embeddings as features for gesture/skill classification
2. **Fine-tune the model**: Focus on gestures that cluster poorly
3. **Analyze confusion**: Study why certain gestures are similar
4. **RSA analysis**: Compare with brain patterns (if using EEG)
5. **Temporal analysis**: Study how embeddings evolve over time within a gesture

For more details, see `src/inference/README_embeddings.md`.
