"""
Visualize embeddings using dimensionality reduction techniques.
"""
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")


def load_embeddings(embedding_path: str) -> Dict[str, np.ndarray]:
    """Load embeddings from pickle file."""
    with open(embedding_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings


def prepare_embeddings_for_visualization(
    embeddings: np.ndarray,
    aggregate_temporal: str = 'mean'
) -> np.ndarray:
    """
    Prepare embeddings for visualization by handling temporal dimension.

    Args:
        embeddings: Embeddings of shape (N, T, D) or (N, D)
        aggregate_temporal: How to aggregate temporal dimension
                          ('mean', 'max', 'first', 'last', None)

    Returns:
        Embeddings of shape (N, D)
    """
    if len(embeddings.shape) == 2:
        # Already (N, D)
        return embeddings
    elif len(embeddings.shape) == 3:
        # (N, T, D) - need to aggregate
        if aggregate_temporal == 'mean':
            return embeddings.mean(axis=1)
        elif aggregate_temporal == 'max':
            return embeddings.max(axis=1)
        elif aggregate_temporal == 'first':
            return embeddings[:, 0, :]
        elif aggregate_temporal == 'last':
            return embeddings[:, -1, :]
        else:
            # Flatten temporal dimension
            N, T, D = embeddings.shape
            return embeddings.reshape(N * T, D)
    else:
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")


def reduce_dimensionality(
    embeddings: np.ndarray,
    method: str = 'tsne',
    n_components: int = 2,
    **kwargs
) -> np.ndarray:
    """
    Reduce dimensionality of embeddings.

    Args:
        embeddings: Embeddings of shape (N, D)
        method: Reduction method ('tsne', 'umap', 'pca')
        n_components: Number of output dimensions
        **kwargs: Additional arguments for the reduction method

    Returns:
        Reduced embeddings of shape (N, n_components)
    """
    print(f"Reducing dimensionality using {method.upper()}...")
    print(f"  Input shape: {embeddings.shape}")

    if method == 'tsne':
        perplexity = kwargs.get('perplexity', min(30, embeddings.shape[0] - 1))
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=kwargs.get('random_state', 42),
            max_iter=kwargs.get('max_iter', 1000)
        )
    elif method == 'umap':
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=kwargs.get('n_neighbors', 15),
            min_dist=kwargs.get('min_dist', 0.1),
            random_state=kwargs.get('random_state', 42)
        )
    elif method == 'pca':
        reducer = PCA(
            n_components=n_components,
            random_state=kwargs.get('random_state', 42)
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    reduced = reducer.fit_transform(embeddings)
    print(f"  Output shape: {reduced.shape}")

    return reduced


def plot_embeddings_2d(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[Dict[int, str]] = None,
    title: str = "Embedding Visualization",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    colormap: str = 'tab20'
):
    """
    Plot 2D embeddings colored by labels.

    Args:
        embeddings_2d: 2D embeddings of shape (N, 2)
        labels: Labels of shape (N,)
        label_names: Optional mapping from label index to name
        title: Plot title
        output_path: Path to save figure (None = show only)
        figsize: Figure size
        colormap: Matplotlib colormap name
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique labels
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)

    # Create color palette
    colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, n_labels))

    # Plot each label
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        label_name = label_names.get(label, f"Label {label}") if label_names else f"Label {label}"

        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[idx]],
            label=label_name,
            alpha=0.7,
            s=50,
            edgecolors='black',
            linewidths=0.5
        )

    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")

    plt.show()


def plot_embeddings_3d(
    embeddings_3d: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[Dict[int, str]] = None,
    title: str = "Embedding Visualization (3D)",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    colormap: str = 'tab20'
):
    """Plot 3D embeddings colored by labels."""
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, n_labels))

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        label_name = label_names.get(label, f"Label {label}") if label_names else f"Label {label}"

        ax.scatter(
            embeddings_3d[mask, 0],
            embeddings_3d[mask, 1],
            embeddings_3d[mask, 2],
            c=[colors[idx]],
            label=label_name,
            alpha=0.7,
            s=50,
            edgecolors='black',
            linewidths=0.5
        )

    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_zlabel('Dimension 3', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")

    plt.show()


def plot_multiple_embeddings(
    all_embeddings: Dict[str, np.ndarray],
    labels: np.ndarray,
    label_names: Optional[Dict[int, str]] = None,
    method: str = 'tsne',
    output_dir: str = 'plots',
    **kwargs
):
    """
    Plot multiple embedding types in a grid.

    Args:
        all_embeddings: Dictionary of embeddings to plot
        labels: Labels for coloring
        label_names: Optional label names
        method: Dimensionality reduction method
        output_dir: Directory to save plots
        **kwargs: Additional arguments for reduction
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_keys = [k for k in all_embeddings.keys()
                     if isinstance(all_embeddings[k], np.ndarray)
                     and not k.endswith('_label')
                     and len(all_embeddings[k].shape) >= 2]

    # Create grid of plots
    n_embeddings = len(embedding_keys)
    n_cols = min(3, n_embeddings)
    n_rows = (n_embeddings + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_embeddings == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, key in enumerate(embedding_keys):
        emb = all_embeddings[key]

        # Prepare embeddings
        emb_2d_input = prepare_embeddings_for_visualization(emb, aggregate_temporal='mean')

        # Reduce dimensionality
        emb_2d = reduce_dimensionality(emb_2d_input, method=method, n_components=2, **kwargs)

        # Plot on subplot
        ax = axes[idx]
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)
        colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, n_labels))

        for label_idx, label in enumerate(unique_labels):
            mask = labels == label
            label_name = label_names.get(label, f"Label {label}") if label_names else f"Label {label}"
            ax.scatter(
                emb_2d[mask, 0],
                emb_2d[mask, 1],
                c=[colors[label_idx]],
                label=label_name,
                alpha=0.7,
                s=30,
                edgecolors='black',
                linewidths=0.5
            )

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(f'{key} ({method.upper()})', fontweight='bold')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Hide empty subplots
    for idx in range(n_embeddings, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    output_path = output_dir / f'embeddings_comparison_{method}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize embeddings')
    parser.add_argument('--embeddings', type=str, required=True,
                       help='Path to embeddings pickle file')
    parser.add_argument('--embedding_type', type=str, default='mean_pooled',
                       help='Which embedding to visualize (mean_pooled, temporal_memory, vit_mid, etc.)')
    parser.add_argument('--label_type', type=str, default='gesture',
                       choices=['gesture', 'skill'],
                       help='Which labels to use for coloring')
    parser.add_argument('--method', type=str, default='tsne',
                       choices=['tsne', 'umap', 'pca'],
                       help='Dimensionality reduction method')
    parser.add_argument('--n_components', type=int, default=2,
                       choices=[2, 3],
                       help='Number of dimensions to reduce to')
    parser.add_argument('--output_dir', type=str, default='plots',
                       help='Output directory for plots')
    parser.add_argument('--aggregate', type=str, default='mean',
                       choices=['mean', 'max', 'first', 'last'],
                       help='How to aggregate temporal dimension')
    parser.add_argument('--compare_all', action='store_true',
                       help='Compare all embedding types in a grid')

    args = parser.parse_args()

    # Load embeddings
    print(f"Loading embeddings from {args.embeddings}...")
    all_embeddings = load_embeddings(args.embeddings)

    print("\nAvailable embeddings:")
    for key, value in all_embeddings.items():
        if isinstance(value, np.ndarray):
            print(f"  - {key}: {value.shape}")

    # Get labels
    label_key = f'{args.label_type}_label'
    if label_key not in all_embeddings:
        raise ValueError(f"Label type '{label_key}' not found in embeddings")

    labels = all_embeddings[label_key]

    # Define label names
    if args.label_type == 'gesture':
        label_names = {i: f'G{i+1}' for i in range(15)}
    else:  # skill
        label_names = {0: 'Novice', 1: 'Intermediate', 2: 'Expert'}

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_all:
        # Compare all embedding types
        print("\nComparing all embedding types...")
        plot_multiple_embeddings(
            all_embeddings,
            labels,
            label_names,
            method=args.method,
            output_dir=output_dir
        )
    else:
        # Visualize single embedding type
        if args.embedding_type not in all_embeddings:
            raise ValueError(f"Embedding type '{args.embedding_type}' not found")

        embeddings = all_embeddings[args.embedding_type]
        print(f"\nVisualizing {args.embedding_type} with shape {embeddings.shape}")

        # Prepare embeddings
        embeddings_prepared = prepare_embeddings_for_visualization(
            embeddings,
            aggregate_temporal=args.aggregate
        )

        # Reduce dimensionality
        embeddings_reduced = reduce_dimensionality(
            embeddings_prepared,
            method=args.method,
            n_components=args.n_components
        )

        # Plot
        title = f"{args.embedding_type} ({args.method.upper()}, colored by {args.label_type})"
        output_path = output_dir / f"{args.embedding_type}_{args.method}_{args.n_components}d.png"

        if args.n_components == 2:
            plot_embeddings_2d(
                embeddings_reduced,
                labels,
                label_names,
                title=title,
                output_path=output_path
            )
        else:  # 3D
            plot_embeddings_3d(
                embeddings_reduced,
                labels,
                label_names,
                title=title,
                output_path=output_path
            )

    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
