"""
Extract embeddings from trained ViT model for visualization and analysis.

Usage:
    # Extract embeddings for a single task on test set
    python src/inference/extract_embeddings.py \
        --checkpoint checkpoints/knot_tying_fold1/best_model.pth \
        --data_root . \
        --task Knot_Tying \
        --split fold_1 \
        --mode test \
        --output embeddings/knot_tying_test_embeddings.pkl

    # Extract embeddings for ALL tasks on test set
    python src/inference/extract_embeddings.py \
        --data_root . \
        --split fold_1 \
        --mode test \
        --all_tasks
"""
import torch
import torch.nn as nn
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_vit_system import EEGInformedViTModel
from data import JIGSAWSViTDataset
from data.split_loader import SplitLoader
from torch.utils.data import DataLoader, Subset


class EmbeddingExtractor:
    """
    Extract embeddings at different levels of the model hierarchy.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Args:
            model: Trained model
            device: Device to run on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def extract_from_batch(
        self,
        batch: Dict[str, torch.Tensor],
        layers: List[str] = ['early', 'mid', 'late']
    ) -> Dict[str, np.ndarray]:
        """
        Extract embeddings from a single batch.

        Args:
            batch: Batch of data
            layers: Which ViT layers to extract ('early', 'mid', 'late')

        Returns:
            Dictionary with extracted embeddings:
            {
                'vit_early': (B, T, D),
                'vit_mid': (B, T, D),
                'vit_late': (B, T, D),
                'cls_tokens': (B, T, D),
                'temporal_memory': (B, T, D),
                'mean_pooled': (B, D),
                'attn_pooled': (B, D),
                'projected': (B, T, D),
                'gesture_label': (B,),
                'skill_label': (B,),
                'trial_id': List[str]
            }
        """
        embeddings = {}

        # Move batch to device
        rgb = batch['rgb'].to(self.device)
        gesture_labels = batch['gesture_label'].cpu().numpy()
        skill_labels = batch['skill_label'].cpu().numpy()
        trial_ids = batch['trial_id']

        with torch.no_grad():
            # Extract ViT layer activations
            emb_layers, cls_tokens = self.model.visual(
                rgb,
                return_layers=layers
            )

            # Store ViT layer embeddings
            for layer_name in layers:
                if layer_name in emb_layers:
                    embeddings[f'vit_{layer_name}'] = emb_layers[layer_name].cpu().numpy()

            # Store CLS tokens
            embeddings['cls_tokens'] = cls_tokens.cpu().numpy()

            # Temporal aggregation
            memory, mean_pooled, attn_pooled = self.model.temporal(cls_tokens)
            embeddings['temporal_memory'] = memory.cpu().numpy()
            embeddings['mean_pooled'] = mean_pooled.cpu().numpy()
            if attn_pooled is not None:
                embeddings['attn_pooled'] = attn_pooled.cpu().numpy()

            # Projected embeddings (pre-decoder)
            # Expand mean_pooled to match temporal dimension
            B, T, D = memory.shape
            pooled_expanded = mean_pooled.unsqueeze(1).expand(B, T, D)
            projected = self.model.kinematics.projection(pooled_expanded)
            embeddings['projected'] = projected.cpu().numpy()

            # Store labels
            embeddings['gesture_label'] = gesture_labels
            embeddings['skill_label'] = skill_labels
            embeddings['trial_id'] = trial_ids

        return embeddings

    def extract_from_dataset(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
        layers: List[str] = ['early', 'mid', 'late']
    ) -> Dict[str, np.ndarray]:
        """
        Extract embeddings from entire dataset.

        Args:
            dataloader: DataLoader for the dataset
            max_batches: Maximum number of batches to process (None = all)
            layers: Which ViT layers to extract

        Returns:
            Dictionary with concatenated embeddings from all batches
        """
        all_embeddings = {
            f'vit_{layer}': [] for layer in layers
        }
        all_embeddings.update({
            'cls_tokens': [],
            'temporal_memory': [],
            'mean_pooled': [],
            'attn_pooled': [],
            'projected': [],
            'gesture_label': [],
            'skill_label': [],
            'trial_id': []
        })

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting embeddings")):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch_embeddings = self.extract_from_batch(batch, layers)

            # Accumulate embeddings
            for key, value in batch_embeddings.items():
                if key in all_embeddings:
                    all_embeddings[key].append(value)

        # Concatenate all batches
        concatenated = {}
        for key, value_list in all_embeddings.items():
            if len(value_list) > 0:
                if key == 'trial_id':
                    # Flatten list of lists
                    concatenated[key] = [item for sublist in value_list for item in sublist]
                else:
                    concatenated[key] = np.concatenate(value_list, axis=0)

        return concatenated


def load_model(checkpoint_path: str, device: torch.device, config_path: str = None) -> nn.Module:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        config_path: Optional path to config file (if not in checkpoint)

    Returns:
        Loaded model
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint or from file
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("Using config from checkpoint")
    elif config_path:
        print(f"Loading config from {config_path}...")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("No config found in checkpoint and no config_path provided")

    print(f"Creating model...")
    model = EEGInformedViTModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown') + 1}")
    return model


def filter_dataset_by_trials(dataset: JIGSAWSViTDataset, trial_ids: List[str]) -> Subset:
    """Filter dataset to only include samples from specific trials."""
    indices = []
    for idx in range(len(dataset)):
        sample_trial = dataset.samples[idx].get('trial_id', '')
        for tid in trial_ids:
            if tid in sample_trial or sample_trial in tid:
                indices.append(idx)
                break
    return Subset(dataset, indices)


def save_embeddings(embeddings: Dict[str, np.ndarray], output_path: str):
    """
    Save embeddings to disk.

    Args:
        embeddings: Dictionary of embeddings
        output_path: Path to save to
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as pickle for easy loading
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"\nEmbeddings saved to {output_path}")
    print(f"Saved embeddings:")
    for key, value in embeddings.items():
        if isinstance(value, np.ndarray):
            print(f"  - {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"  - {key}: {len(value)} items")


def extract_embeddings_for_task(
    task: str,
    checkpoint_path: Path,
    data_root: Path,
    output_path: Path,
    split: str,
    mode: str,
    device: torch.device,
    batch_size: int = 8,
    layers: List[str] = ['early', 'mid', 'late'],
    config_path: str = None
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings for a single task.

    Args:
        task: Task name (e.g., 'Knot_Tying')
        checkpoint_path: Path to model checkpoint
        data_root: Path to data root directory
        output_path: Path to save embeddings
        split: Split name (e.g., 'fold_1')
        mode: Which set to use ('train', 'val', 'test')
        device: Device to run on
        batch_size: Batch size for extraction
        layers: Which ViT layers to extract
        config_path: Optional path to config file

    Returns:
        Dictionary of extracted embeddings
    """
    print(f"\n{'='*60}")
    print(f"Extracting embeddings for {task}")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {split}, Mode: {mode}")

    # Load model
    model = load_model(str(checkpoint_path), device, config_path)

    # Create full dataset
    print(f"\nLoading dataset...")
    full_dataset = JIGSAWSViTDataset(
        data_root=str(data_root),
        task=task,
        mode='train',
        arm='PSM2'
    )
    print(f"Full dataset: {len(full_dataset)} samples")

    # Filter by split
    if split:
        split_loader = SplitLoader(str(data_root), task, split)

        if mode == 'test':
            trial_ids = split_loader.get_test_trials()
        elif mode == 'val':
            trial_ids = split_loader.get_val_trials()
        else:
            trial_ids = split_loader.get_train_trials()

        print(f"Filtering to {mode} set trials: {trial_ids}")
        dataset = filter_dataset_by_trials(full_dataset, trial_ids)
        print(f"Filtered dataset: {len(dataset)} samples")
    else:
        dataset = full_dataset

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Extract embeddings
    print(f"\nExtracting embeddings from layers: {layers}")
    extractor = EmbeddingExtractor(model, device)
    embeddings = extractor.extract_from_dataset(dataloader, layers=layers)

    # Save embeddings
    save_embeddings(embeddings, str(output_path))

    return embeddings


def main():
    parser = argparse.ArgumentParser(description='Extract embeddings from trained model')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (required if not using --all_tasks)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (optional, uses config from checkpoint)')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to data root directory')
    parser.add_argument('--task', type=str, default='Knot_Tying',
                       help='Task name (ignored if --all_tasks)')
    parser.add_argument('--split', type=str, default='fold_1',
                       help='Split name (e.g., fold_1)')
    parser.add_argument('--mode', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Which set to extract embeddings from')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for embeddings (auto-generated if not specified)')
    parser.add_argument('--output_dir', type=str, default='embeddings',
                       help='Output directory for embeddings')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for extraction')
    parser.add_argument('--layers', type=str, nargs='+',
                       default=['early', 'mid', 'late'],
                       help='Which ViT layers to extract')
    parser.add_argument('--all_tasks', action='store_true',
                       help='Extract embeddings for all three tasks')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Base directory for checkpoints (used with --all_tasks)')

    args = parser.parse_args()

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using device: MPS")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using device: CUDA")
    else:
        device = torch.device('cpu')
        print("Using device: CPU")

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_tasks:
        # Extract for all three tasks
        tasks = ['Knot_Tying', 'Needle_Passing', 'Suturing']
        checkpoint_dir = Path(args.checkpoint_dir)

        print("="*60)
        print("Extracting embeddings for ALL tasks")
        print(f"Split: {args.split}, Mode: {args.mode}")
        print("="*60)

        for task in tasks:
            task_lower = task.lower()
            checkpoint_path = checkpoint_dir / f"{task_lower}_{args.split}" / "best_model.pth"

            if not checkpoint_path.exists():
                print(f"\nWARNING: Checkpoint not found for {task}: {checkpoint_path}")
                print("Skipping...")
                continue

            output_path = output_dir / f"{task_lower}_{args.mode}_{args.split}_embeddings.pkl"

            extract_embeddings_for_task(
                task=task,
                checkpoint_path=checkpoint_path,
                data_root=data_root,
                output_path=output_path,
                split=args.split,
                mode=args.mode,
                device=device,
                batch_size=args.batch_size,
                layers=args.layers,
                config_path=args.config
            )

        print("\n" + "="*60)
        print("All extractions complete!")
        print("="*60)
        print(f"\nEmbeddings saved to: {output_dir}")
        for task in tasks:
            task_lower = task.lower()
            print(f"  - {task_lower}_{args.mode}_{args.split}_embeddings.pkl")

    else:
        # Single task extraction
        if args.checkpoint is None:
            parser.error("--checkpoint is required when not using --all_tasks")

        checkpoint_path = Path(args.checkpoint)

        if args.output:
            output_path = Path(args.output)
        else:
            task_lower = args.task.lower()
            output_path = output_dir / f"{task_lower}_{args.mode}_{args.split}_embeddings.pkl"

        extract_embeddings_for_task(
            task=args.task,
            checkpoint_path=checkpoint_path,
            data_root=data_root,
            output_path=output_path,
            split=args.split,
            mode=args.mode,
            device=device,
            batch_size=args.batch_size,
            layers=args.layers,
            config_path=args.config
        )

    print("\nTo visualize embeddings, run:")
    print(f"  python src/inference/visualize_embeddings.py --embeddings {output_dir}/<task>_embeddings.pkl")


if __name__ == '__main__':
    main()
