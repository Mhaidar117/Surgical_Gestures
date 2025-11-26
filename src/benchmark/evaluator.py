"""
Benchmark evaluator for model checkpoints.
"""
import torch
import torch.nn as nn
import yaml
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.train_vit_system import EEGInformedViTModel
from src.data import JIGSAWSViTDataset
from src.eval.metrics import (
    compute_kinematics_metrics_detailed,
    compute_gesture_metrics_detailed,
    compute_skill_metrics_detailed
)
from torch.utils.data import DataLoader


class BenchmarkEvaluator:
    """
    Evaluator for benchmarking saved model checkpoints.
    """

    def __init__(
        self,
        checkpoint_path: str,
        data_root: str,
        task: str = 'Knot_Tying',
        mode: str = 'val',
        device: str = 'auto',
        batch_size: int = 16,
        num_workers: int = 4
    ):
        """
        Initialize evaluator.

        Args:
            checkpoint_path: Path to saved checkpoint
            data_root: Root directory containing Gestures/ folder
            task: Task name ('Knot_Tying', 'Needle_Passing', 'Suturing')
            mode: Data split mode ('train', 'val', 'test')
            device: Device to use ('auto', 'cuda', 'mps', 'cpu')
            batch_size: Batch size for evaluation
            num_workers: Number of dataloader workers
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.data_root = Path(data_root)
        self.task = task
        self.mode = mode
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Auto-select device
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Will be set by load_model()
        self.model = None
        self.config = None
        self.dataloader = None

    def load_model(self) -> None:
        """Load model from checkpoint."""
        print(f"Loading checkpoint from {self.checkpoint_path}...")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Extract config
        if 'config' not in checkpoint:
            raise ValueError("Checkpoint does not contain 'config' key")

        self.config = checkpoint['config']

        # Create model
        print("Creating model from config...")
        self.model = EEGInformedViTModel(self.config)

        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully")

        # Print checkpoint info
        if 'epoch' in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch'] + 1}")
        if 'losses' in checkpoint:
            print(f"Training loss: {checkpoint['losses'].get('total', 'N/A'):.4f}")

    def create_dataloader(self) -> None:
        """Create dataloader for evaluation."""
        print(f"Creating dataloader for task '{self.task}', mode '{self.mode}'...")

        # Create dataset
        dataset = JIGSAWSViTDataset(
            data_root=str(self.data_root),
            task=self.task,
            mode=self.mode,
            use_rgb=True,
            use_flow=self.config.get('use_flow', False),
            use_eeg=False
        )

        print(f"Dataset size: {len(dataset)} samples")

        # Create dataloader
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Don't shuffle for evaluation
            num_workers=0,  # Set to 0 for compatibility
            drop_last=False
        )

        print(f"Dataloader created with {len(self.dataloader)} batches")

    @torch.no_grad()
    def run_evaluation(self) -> Tuple[Dict, Dict]:
        """
        Run evaluation on the dataset.

        Returns:
            Tuple of (predictions, targets) dictionaries
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first")
        if self.dataloader is None:
            raise ValueError("Dataloader not created. Call create_dataloader() first")

        print("\nRunning evaluation...")

        # Collect predictions and targets
        all_kin_pred = []
        all_kin_target = []
        all_gesture_logits = []
        all_gesture_labels = []
        all_skill_logits = []
        all_skill_labels = []

        # Inference loop
        for batch in tqdm(self.dataloader, desc="Evaluating"):
            # Move to device
            rgb = batch['rgb'].to(self.device)
            kinematics = batch['kinematics'].to(self.device)
            gesture_labels = batch['gesture_label'].to(self.device)
            skill_labels = batch['skill_label'].to(self.device)

            # Forward pass (no teacher forcing)
            outputs = self.model(
                rgb,
                target_kinematics=None,
                teacher_forcing_prob=0.0
            )

            # Collect predictions (move to CPU immediately to save GPU memory)
            all_kin_pred.append(outputs['kinematics'].cpu())
            all_kin_target.append(kinematics[:, :, :10].cpu())  # Match output dimensions
            all_gesture_logits.append(outputs['gesture_logits'].cpu())
            all_gesture_labels.append(gesture_labels.cpu())
            all_skill_logits.append(outputs['skill_logits'].cpu())
            all_skill_labels.append(skill_labels.cpu())

        # Concatenate all batches
        print("Aggregating results...")
        predictions = {
            'kinematics': torch.cat(all_kin_pred, dim=0),
            'gesture_logits': torch.cat(all_gesture_logits, dim=0),
            'skill_logits': torch.cat(all_skill_logits, dim=0)
        }

        targets = {
            'kinematics': torch.cat(all_kin_target, dim=0),
            'gesture_labels': torch.cat(all_gesture_labels, dim=0),
            'skill_labels': torch.cat(all_skill_labels, dim=0)
        }

        print(f"Evaluation complete. Processed {predictions['kinematics'].shape[0]} samples")

        return predictions, targets

    def compute_metrics(self, predictions: Dict, targets: Dict) -> Dict:
        """
        Compute all metrics from predictions and targets.

        Args:
            predictions: Dictionary with model predictions
            targets: Dictionary with ground truth targets

        Returns:
            Dictionary with all computed metrics
        """
        print("\nComputing metrics...")

        metrics = {}

        # Kinematics metrics
        print("  - Kinematics metrics...")
        kin_metrics = compute_kinematics_metrics_detailed(
            predictions['kinematics'],
            targets['kinematics']
        )
        metrics['kinematics'] = kin_metrics

        # Gesture metrics
        print("  - Gesture classification metrics...")
        gesture_metrics = compute_gesture_metrics_detailed(
            predictions['gesture_logits'],
            targets['gesture_labels']
        )
        metrics['gesture'] = gesture_metrics

        # Skill metrics
        print("  - Skill classification metrics...")
        skill_metrics = compute_skill_metrics_detailed(
            predictions['skill_logits'],
            targets['skill_labels']
        )
        metrics['skill'] = skill_metrics

        print("Metrics computation complete")

        return metrics

    def save_results(
        self,
        metrics: Dict,
        predictions: Dict,
        targets: Dict,
        output_dir: Path
    ) -> None:
        """
        Save evaluation results to output directory.

        Args:
            metrics: Computed metrics
            predictions: Model predictions
            targets: Ground truth targets
            output_dir: Output directory path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving results to {output_dir}...")

        # Create subdirectories
        metrics_dir = output_dir / 'metrics'
        metrics_dir.mkdir(exist_ok=True)

        # Save metrics as JSON (convert numpy arrays to lists)
        metrics_json = {}
        for task, task_metrics in metrics.items():
            metrics_json[task] = {}
            for key, value in task_metrics.items():
                if isinstance(value, dict):
                    # Nested dict (e.g., per-class metrics)
                    metrics_json[task][key] = value
                elif hasattr(value, 'tolist'):
                    # Numpy array
                    metrics_json[task][key] = value.tolist()
                else:
                    # Scalar
                    metrics_json[task][key] = value

        metrics_file = metrics_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        print(f"  - Saved metrics to {metrics_file}")

        # Save checkpoint info
        checkpoint_info = {
            'checkpoint_path': str(self.checkpoint_path),
            'task': self.task,
            'mode': self.mode,
            'device': str(self.device),
            'num_samples': predictions['kinematics'].shape[0]
        }

        info_file = output_dir / 'checkpoint_info.json'
        with open(info_file, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
        print(f"  - Saved checkpoint info to {info_file}")

        # Save config
        config_file = output_dir / 'config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f)
        print(f"  - Saved config to {config_file}")

        print("Results saved successfully")

    def run_full_evaluation(self, output_dir: Path) -> Dict:
        """
        Run complete evaluation pipeline.

        Args:
            output_dir: Output directory for results

        Returns:
            Dictionary with all metrics
        """
        # Load model
        self.load_model()

        # Create dataloader
        self.create_dataloader()

        # Run evaluation
        predictions, targets = self.run_evaluation()

        # Compute metrics
        metrics = self.compute_metrics(predictions, targets)

        # Save results
        self.save_results(metrics, predictions, targets, output_dir)

        # Return metrics and predictions for visualization
        return {
            'metrics': metrics,
            'predictions': predictions,
            'targets': targets
        }
