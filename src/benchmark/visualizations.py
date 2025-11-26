"""
Visualization functions for benchmark evaluation.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from mpl_toolkits.mplot3d import Axes3D


class VisualizationManager:
    """Manager for creating all benchmark visualizations."""

    def __init__(
        self,
        output_dir: Path,
        dpi: int = 300,
        format: str = 'png'
    ):
        """
        Initialize visualization manager.

        Args:
            output_dir: Directory to save visualizations
            dpi: DPI for saved figures
            format: Image format ('png', 'pdf', 'svg')
        """
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / 'visualizations'
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.format = format

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = dpi

    def save_figure(self, fig: plt.Figure, filename: str) -> Path:
        """Save figure to file."""
        filepath = self.viz_dir / f"{filename}.{self.format}"
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        return filepath

    def generate_all_plots(
        self,
        predictions: Dict,
        targets: Dict,
        metrics: Dict
    ) -> Dict[str, Path]:
        """
        Generate all visualization plots.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            metrics: Computed metrics

        Returns:
            Dictionary mapping plot names to file paths
        """
        print("\nGenerating visualizations...")
        plot_paths = {}

        # Gesture confusion matrix
        print("  - Gesture confusion matrix...")
        path = self.plot_confusion_matrix(
            metrics['gesture']['confusion_matrix'],
            class_names=[f'G{i+1}' for i in range(15)],
            title='Gesture Classification Confusion Matrix',
            filename='gesture_confusion_matrix'
        )
        plot_paths['gesture_confusion'] = path

        # Skill confusion matrix
        print("  - Skill confusion matrix...")
        path = self.plot_confusion_matrix(
            metrics['skill']['confusion_matrix'],
            class_names=['Novice', 'Intermediate', 'Expert'],
            title='Skill Classification Confusion Matrix',
            filename='skill_confusion_matrix'
        )
        plot_paths['skill_confusion'] = path

        # Per-class gesture metrics
        print("  - Per-class gesture metrics...")
        path = self.plot_per_class_metrics(
            metrics['gesture'],
            title='Per-Class Gesture Performance',
            filename='per_class_gesture_metrics'
        )
        plot_paths['per_class_gesture'] = path

        # Trajectory samples
        print("  - Trajectory samples...")
        num_samples = min(5, predictions['kinematics'].shape[0])
        for i in range(num_samples):
            path = self.plot_trajectory_comparison(
                predictions['kinematics'][i, :, :3].numpy(),
                targets['kinematics'][i, :, :3].numpy(),
                filename=f'trajectory_sample_{i}'
            )
            plot_paths[f'trajectory_{i}'] = path

        # Error distributions
        print("  - Error distributions...")
        path = self.plot_error_distribution(
            predictions['kinematics'],
            targets['kinematics'],
            filename='position_error_distribution'
        )
        plot_paths['position_error_dist'] = path

        # Kinematics time series
        print("  - Kinematics time series...")
        path = self.plot_kinematics_components(
            predictions['kinematics'][0].numpy(),
            targets['kinematics'][0].numpy(),
            filename='kinematics_time_series_sample_0'
        )
        plot_paths['kinematics_time_series'] = path

        print("Visualizations complete")
        return plot_paths

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        title: str,
        filename: str
    ) -> Path:
        """
        Plot confusion matrix as heatmap.

        Args:
            cm: Confusion matrix (n_classes, n_classes)
            class_names: List of class names
            title: Plot title
            filename: Output filename (without extension)

        Returns:
            Path to saved figure
        """
        # Normalize by row (true class)
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={'label': 'Normalized Frequency'}
        )

        ax.set_title(title, fontsize=14, pad=15)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        return self.save_figure(fig, filename)

    def plot_trajectory_comparison(
        self,
        pred_traj: np.ndarray,
        target_traj: np.ndarray,
        filename: str
    ) -> Path:
        """
        Plot 3D trajectory comparison.

        Args:
            pred_traj: Predicted trajectory (T, 3)
            target_traj: Target trajectory (T, 3)
            filename: Output filename

        Returns:
            Path to saved figure
        """
        fig = plt.figure(figsize=(15, 4))

        # 3D plot
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(
            target_traj[:, 0], target_traj[:, 1], target_traj[:, 2],
            'b-', label='Ground Truth', linewidth=2
        )
        ax1.plot(
            pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2],
            'r--', label='Predicted', linewidth=2
        )
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D Trajectory')
        ax1.legend()
        ax1.grid(True)

        # XY projection
        ax2 = fig.add_subplot(132)
        ax2.plot(target_traj[:, 0], target_traj[:, 1], 'b-', label='Ground Truth', linewidth=2)
        ax2.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', label='Predicted', linewidth=2)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('XY Projection')
        ax2.legend()
        ax2.grid(True)
        ax2.axis('equal')

        # XZ projection
        ax3 = fig.add_subplot(133)
        ax3.plot(target_traj[:, 0], target_traj[:, 2], 'b-', label='Ground Truth', linewidth=2)
        ax3.plot(pred_traj[:, 0], pred_traj[:, 2], 'r--', label='Predicted', linewidth=2)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.set_title('XZ Projection')
        ax3.legend()
        ax3.grid(True)
        ax3.axis('equal')

        plt.tight_layout()
        return self.save_figure(fig, filename)

    def plot_error_distribution(
        self,
        pred_kinematics: torch.Tensor,
        target_kinematics: torch.Tensor,
        filename: str
    ) -> Path:
        """
        Plot error distribution for position errors.

        Args:
            pred_kinematics: Predicted kinematics (B, T, 10)
            target_kinematics: Target kinematics (B, T, 10)
            filename: Output filename

        Returns:
            Path to saved figure
        """
        # Compute position errors
        pos_pred = pred_kinematics[:, :, :3].numpy()
        pos_target = target_kinematics[:, :, :3].numpy()

        # Euclidean distance per frame
        errors = np.linalg.norm(pos_pred - pos_target, axis=-1).flatten()

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram
        ax1.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(errors.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.4f}')
        ax1.axvline(np.median(errors), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.4f}')
        ax1.set_xlabel('Position Error (m)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Position Error Distribution', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(errors, vert=True)
        ax2.set_ylabel('Position Error (m)', fontsize=12)
        ax2.set_title('Position Error Box Plot', fontsize=14)
        ax2.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = f'Mean: {errors.mean():.4f}\n'
        stats_text += f'Std: {errors.std():.4f}\n'
        stats_text += f'Max: {errors.max():.4f}\n'
        stats_text += f'Min: {errors.min():.4f}'
        ax2.text(1.15, 0.5, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        return self.save_figure(fig, filename)

    def plot_per_class_metrics(
        self,
        gesture_metrics: Dict,
        title: str,
        filename: str
    ) -> Path:
        """
        Plot per-class precision, recall, and F1 scores.

        Args:
            gesture_metrics: Gesture metrics dictionary
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved figure
        """
        # Extract per-class metrics
        precision = gesture_metrics['precision_per_class']
        recall = gesture_metrics['recall_per_class']
        f1 = gesture_metrics['f1_per_class']

        # Prepare data
        classes = list(precision.keys())
        precision_vals = [precision[c] for c in classes]
        recall_vals = [recall[c] for c in classes]
        f1_vals = [f1[c] for c in classes]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(classes))
        width = 0.25

        # Plot bars
        ax.bar(x - width, precision_vals, width, label='Precision', alpha=0.8)
        ax.bar(x, recall_vals, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_vals, width, label='F1-Score', alpha=0.8)

        # Customize plot
        ax.set_xlabel('Gesture Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        return self.save_figure(fig, filename)

    def plot_kinematics_components(
        self,
        pred_kin: np.ndarray,
        target_kin: np.ndarray,
        filename: str
    ) -> Path:
        """
        Plot time series of kinematics components.

        Args:
            pred_kin: Predicted kinematics (T, 10)
            target_kin: Target kinematics (T, 10)
            filename: Output filename

        Returns:
            Path to saved figure
        """
        T = pred_kin.shape[0]
        time_steps = np.arange(T)

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # Position X, Y, Z
        for i, axis_name in enumerate(['X', 'Y', 'Z']):
            axes[i].plot(time_steps, target_kin[:, i], 'b-', label='Ground Truth', linewidth=2)
            axes[i].plot(time_steps, pred_kin[:, i], 'r--', label='Predicted', linewidth=2)
            axes[i].set_xlabel('Time Step', fontsize=10)
            axes[i].set_ylabel(f'Position {axis_name} (m)', fontsize=10)
            axes[i].set_title(f'Position {axis_name}', fontsize=12)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        # Gripper/jaw
        if pred_kin.shape[-1] >= 10:
            axes[3].plot(time_steps, target_kin[:, 9], 'b-', label='Ground Truth', linewidth=2)
            axes[3].plot(time_steps, pred_kin[:, 9], 'r--', label='Predicted', linewidth=2)
            axes[3].set_xlabel('Time Step', fontsize=10)
            axes[3].set_ylabel('Gripper Angle (rad)', fontsize=10)
            axes[3].set_title('Gripper/Jaw', fontsize=12)
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        return self.save_figure(fig, filename)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: Path,
    dpi: int = 300
) -> None:
    """
    Standalone function to plot confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: Class names
        title: Plot title
        save_path: Path to save figure
        dpi: DPI for saved figure
    """
    # Normalize
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)

    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
