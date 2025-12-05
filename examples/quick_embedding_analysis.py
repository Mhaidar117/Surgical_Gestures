"""
Quick embedding analysis example - one script to extract and visualize.
Usage: python examples/quick_embedding_analysis.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from training.train_vit_system import EEGInformedViTModel
from data import JIGSAWSViTDataset
from torch.utils.data import DataLoader


def quick_analysis(
    checkpoint_path: str,
    config_path: str,
    data_root: str,
    task: str = 'Knot_Tying',
    num_samples: int = 100
):
    """
    Quick embedding extraction and visualization.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        data_root: Path to JIGSAWS data
        task: Task name
        num_samples: Number of samples to analyze
    """
    print("=" * 60)
    print("QUICK EMBEDDING ANALYSIS")
    print("=" * 60)

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = EEGInformedViTModel(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model loaded (epoch {checkpoint.get('epoch', '?')})")

    # Load dataset
    print(f"\nLoading dataset: {task}...")
    dataset = JIGSAWSViTDataset(
        data_root=data_root,
        task=task,
        mode='train'
    )

    # Limit samples
    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset = torch.utils.data.Subset(dataset, indices)

    dataloader = DataLoader(subset, batch_size=8, shuffle=False, num_workers=0)
    print(f"✓ Loaded {num_samples} samples")

    # Extract embeddings
    print("\nExtracting embeddings...")
    all_embeddings = []
    all_gesture_labels = []
    all_skill_labels = []

    with torch.no_grad():
        for batch in dataloader:
            rgb = batch['rgb'].to(device)
            gesture_labels = batch['gesture_label'].cpu().numpy()
            skill_labels = batch['skill_label'].cpu().numpy()

            # Extract from temporal transformer (contextualized representation)
            emb_layers, cls_tokens = model.visual(rgb)
            memory, mean_pooled, attn_pooled = model.temporal(cls_tokens)

            # Use mean pooled (trial-level representation)
            embeddings = mean_pooled.cpu().numpy()

            all_embeddings.append(embeddings)
            all_gesture_labels.append(gesture_labels)
            all_skill_labels.append(skill_labels)

    # Concatenate
    embeddings = np.concatenate(all_embeddings, axis=0)  # (N, 384)
    gesture_labels = np.concatenate(all_gesture_labels, axis=0)  # (N,)
    skill_labels = np.concatenate(all_skill_labels, axis=0)  # (N,)

    print(f"✓ Extracted embeddings: {embeddings.shape}")
    print(f"  - Unique gestures: {len(np.unique(gesture_labels))}")
    print(f"  - Unique skills: {len(np.unique(skill_labels))}")

    # Visualize
    print("\nGenerating visualizations...")

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Embedding Analysis: {task}', fontsize=16, fontweight='bold')

    # 1. t-SNE colored by gesture
    print("  - t-SNE (gesture)...")
    tsne = TSNE(n_components=2, perplexity=min(30, num_samples - 1), random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    ax = axes[0, 0]
    for gesture in np.unique(gesture_labels):
        mask = gesture_labels == gesture
        ax.scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1],
                  label=f'G{gesture+1}', alpha=0.7, s=50)
    ax.set_title('t-SNE: Colored by Gesture', fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. t-SNE colored by skill
    print("  - t-SNE (skill)...")
    skill_names = {0: 'Novice', 1: 'Intermediate', 2: 'Expert'}
    skill_colors = {0: 'red', 1: 'orange', 2: 'green'}

    ax = axes[0, 1]
    for skill in np.unique(skill_labels):
        mask = skill_labels == skill
        ax.scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1],
                  label=skill_names[skill], alpha=0.7, s=50,
                  c=skill_colors[skill])
    ax.set_title('t-SNE: Colored by Skill Level', fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. PCA colored by gesture
    print("  - PCA (gesture)...")
    pca = PCA(n_components=2, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)

    ax = axes[1, 0]
    for gesture in np.unique(gesture_labels):
        mask = gesture_labels == gesture
        ax.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1],
                  label=f'G{gesture+1}', alpha=0.7, s=50)
    ax.set_title(f'PCA: Colored by Gesture\n(Explained Var: {pca.explained_variance_ratio_.sum():.2%})',
                fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. PCA colored by skill
    print("  - PCA (skill)...")
    ax = axes[1, 1]
    for skill in np.unique(skill_labels):
        mask = skill_labels == skill
        ax.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1],
                  label=skill_names[skill], alpha=0.7, s=50,
                  c=skill_colors[skill])
    ax.set_title('PCA: Colored by Skill Level', fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = f'quick_analysis_{task}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_path}")

    plt.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total samples analyzed: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"\nGesture distribution:")
    for gesture in np.unique(gesture_labels):
        count = np.sum(gesture_labels == gesture)
        print(f"  G{gesture+1}: {count} samples ({100*count/len(gesture_labels):.1f}%)")
    print(f"\nSkill distribution:")
    for skill in np.unique(skill_labels):
        count = np.sum(skill_labels == skill)
        print(f"  {skill_names[skill]}: {count} samples ({100*count/len(skill_labels):.1f}%)")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    # EDIT THESE PATHS
    checkpoint_path = 'checkpoints/baseline_psm2_needle_passing_3/checkpoint_epoch_40.pth'#'checkpoints/baseline/checkpoint_epoch_10.pth'
    config_path = './src/configs/baseline.yaml'
    data_root = 'Gestures/Needle_Passing/video/Needle_Passing_B001_capture1.avi'  # UPDATE THIS

    # Run quick analysis
    quick_analysis(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        data_root=data_root,
        task='Needle_Passing',
        num_samples=100  # Adjust based on your dataset size
    )
