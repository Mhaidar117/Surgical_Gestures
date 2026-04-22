"""
Analyze similarity between specific gestures using embeddings.
Useful for understanding which gestures are easily confused.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine, euclidean
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics.pairwise import cosine_similarity


def compute_gesture_centroids(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute centroid embedding for each gesture.

    Args:
        embeddings: (N, D) embeddings
        labels: (N,) gesture labels

    Returns:
        Dictionary mapping gesture label to centroid
    """
    centroids = {}
    for gesture in np.unique(labels):
        mask = labels == gesture
        centroids[gesture] = embeddings[mask].mean(axis=0)
    return centroids


def compute_similarity_matrix(centroids: dict, metric: str = 'cosine') -> np.ndarray:
    """
    Compute pairwise similarity between gesture centroids.

    Args:
        centroids: Dictionary of gesture centroids
        metric: 'cosine' or 'euclidean'

    Returns:
        Similarity matrix (num_gestures, num_gestures)
    """
    gestures = sorted(centroids.keys())
    n = len(gestures)
    similarity_matrix = np.zeros((n, n))

    for i, g1 in enumerate(gestures):
        for j, g2 in enumerate(gestures):
            if metric == 'cosine':
                # Cosine similarity (higher = more similar)
                sim = 1 - cosine(centroids[g1], centroids[g2])
            else:  # euclidean
                # Negative euclidean distance (higher = more similar)
                sim = -euclidean(centroids[g1], centroids[g2])
            similarity_matrix[i, j] = sim

    return similarity_matrix


def plot_similarity_matrix(similarity_matrix: np.ndarray, gesture_names: list, title: str = ''):
    """Plot similarity matrix as heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity_matrix,
        xticklabels=gesture_names,
        yticklabels=gesture_names,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=0,
        square=True,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title(title, fontweight='bold', fontsize=14)
    plt.xlabel('Gesture', fontsize=12)
    plt.ylabel('Gesture', fontsize=12)
    plt.tight_layout()


def plot_dendrogram(centroids: dict, gesture_names: list):
    """Plot hierarchical clustering dendrogram."""
    # Create linkage matrix
    centroid_array = np.array([centroids[g] for g in sorted(centroids.keys())])
    linkage_matrix = linkage(centroid_array, method='ward')

    plt.figure(figsize=(12, 6))
    dendrogram(
        linkage_matrix,
        labels=gesture_names,
        leaf_font_size=10
    )
    plt.title('Hierarchical Clustering of Gestures', fontweight='bold', fontsize=14)
    plt.xlabel('Gesture', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.tight_layout()


def find_most_similar_pairs(similarity_matrix: np.ndarray, gesture_names: list, top_k: int = 5):
    """Find most similar gesture pairs."""
    n = len(gesture_names)
    pairs = []

    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j, similarity_matrix[i, j]))

    # Sort by similarity (descending)
    pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'='*60}")
    print(f"TOP {top_k} MOST SIMILAR GESTURE PAIRS")
    print(f"{'='*60}")
    for idx, (i, j, sim) in enumerate(pairs[:top_k], 1):
        print(f"{idx}. {gesture_names[i]} ↔ {gesture_names[j]}: {sim:.4f}")


def find_most_dissimilar_pairs(similarity_matrix: np.ndarray, gesture_names: list, top_k: int = 5):
    """Find most dissimilar gesture pairs."""
    n = len(gesture_names)
    pairs = []

    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j, similarity_matrix[i, j]))

    # Sort by similarity (ascending)
    pairs.sort(key=lambda x: x[2])

    print(f"\n{'='*60}")
    print(f"TOP {top_k} MOST DISSIMILAR GESTURE PAIRS")
    print(f"{'='*60}")
    for idx, (i, j, sim) in enumerate(pairs[:top_k], 1):
        print(f"{idx}. {gesture_names[i]} ↔ {gesture_names[j]}: {sim:.4f}")


def analyze_gesture_variance(embeddings: np.ndarray, labels: np.ndarray):
    """Analyze within-gesture variance."""
    print(f"\n{'='*60}")
    print("WITHIN-GESTURE VARIANCE (lower = more consistent)")
    print(f"{'='*60}")

    variances = []
    for gesture in sorted(np.unique(labels)):
        mask = labels == gesture
        gesture_embeddings = embeddings[mask]

        # Compute variance (mean distance to centroid)
        centroid = gesture_embeddings.mean(axis=0)
        distances = np.linalg.norm(gesture_embeddings - centroid, axis=1)
        variance = distances.mean()

        variances.append((gesture, variance, len(gesture_embeddings)))

    # Sort by variance
    variances.sort(key=lambda x: x[1])

    for gesture, var, count in variances:
        print(f"G{gesture+1}: {var:.4f} (n={count})")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze gesture similarity')
    parser.add_argument('--embeddings', type=str, required=True,
                       help='Path to embeddings pickle file')
    parser.add_argument('--embedding_type', type=str, default='mean_pooled',
                       help='Which embedding to analyze')
    parser.add_argument('--metric', type=str, default='cosine',
                       choices=['cosine', 'euclidean'],
                       help='Distance metric')
    parser.add_argument('--output_dir', type=str, default='plots/similarity',
                       help='Output directory')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top pairs to show')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    print(f"Loading embeddings from {args.embeddings}...")
    with open(args.embeddings, 'rb') as f:
        data = pickle.load(f)

    if args.embedding_type not in data:
        print(f"Error: {args.embedding_type} not found in embeddings")
        print(f"Available: {[k for k in data.keys() if isinstance(data[k], np.ndarray)]}")
        return

    embeddings = data[args.embedding_type]
    gesture_labels = data['gesture_label']

    # Handle temporal dimension
    if len(embeddings.shape) == 3:
        print(f"Aggregating temporal dimension (mean)...")
        embeddings = embeddings.mean(axis=1)

    print(f"Analyzing {len(embeddings)} samples with {len(np.unique(gesture_labels))} gestures")

    # Compute centroids
    print("\nComputing gesture centroids...")
    centroids = compute_gesture_centroids(embeddings, gesture_labels)

    # Compute similarity matrix
    print(f"Computing {args.metric} similarity matrix...")
    similarity_matrix = compute_similarity_matrix(centroids, metric=args.metric)

    # Gesture names
    gesture_names = [f'G{i+1}' for i in sorted(centroids.keys())]

    # Plot similarity matrix
    print("\nGenerating similarity heatmap...")
    plot_similarity_matrix(
        similarity_matrix,
        gesture_names,
        title=f'Gesture Similarity Matrix ({args.metric.capitalize()})'
    )
    plt.savefig(output_dir / f'similarity_matrix_{args.metric}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot dendrogram
    print("Generating dendrogram...")
    plot_dendrogram(centroids, gesture_names)
    plt.savefig(output_dir / 'dendrogram.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Find similar/dissimilar pairs
    find_most_similar_pairs(similarity_matrix, gesture_names, args.top_k)
    find_most_dissimilar_pairs(similarity_matrix, gesture_names, args.top_k)

    # Analyze variance
    analyze_gesture_variance(embeddings, gesture_labels)

    print(f"\n{'='*60}")
    print(f"Analysis complete! Plots saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
