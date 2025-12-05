"""
Evaluate a trained model on validation/test set.

Usage:
    python src/eval/evaluate.py \
        --checkpoint checkpoints/baseline_psm2_knot_tying/checkpoint_epoch_50.pth \
        --data_root . \
        --task Knot_Tying \
        --split fold_1 \
        --mode val \
        --output_dir eval_results
"""
import torch
import torch.nn as nn
import yaml
import argparse
from pathlib import Path
from typing import Dict, Optional, List
import sys
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.losses import compute_total_loss
from eval.metrics import compute_kinematics_metrics, compute_gesture_metrics, compute_skill_metrics
from data import JIGSAWSViTDataset
from data.split_loader import SplitLoader
from torch.utils.data import DataLoader, Subset


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    from training.train_vit_system import EEGInformedViTModel

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = EEGInformedViTModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, config, checkpoint.get('epoch', -1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: Dict
) -> Dict[str, float]:
    """Evaluate model on a dataset."""
    model.eval()

    all_losses = {}
    all_kin_preds = []
    all_kin_targets = []
    all_gesture_logits = []
    all_gesture_labels = []
    all_skill_logits = []
    all_skill_labels = []

    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            # Move to device
            rgb = batch['rgb'].to(device)
            kinematics = batch['kinematics'].to(device)
            gesture_labels = batch['gesture_label'].to(device)
            skill_labels = batch['skill_label'].to(device)

            # Forward pass (no teacher forcing during evaluation)
            outputs = model(
                rgb,
                target_kinematics=kinematics,
                teacher_forcing_prob=0.0  # No teacher forcing
            )

            # Compute loss
            loss, losses = compute_total_loss(
                outputs['kinematics'],
                kinematics,
                outputs['gesture_logits'],
                gesture_labels,
                outputs['skill_logits'],
                skill_labels,
                loss_weights=config.get('loss_weights')
            )

            # Accumulate losses
            for k, v in losses.items():
                if k not in all_losses:
                    all_losses[k] = 0.0
                all_losses[k] += v.item() if isinstance(v, torch.Tensor) else v

            # Collect predictions for metrics
            all_kin_preds.append(outputs['kinematics'].cpu())
            all_kin_targets.append(kinematics.cpu())
            all_gesture_logits.append(outputs['gesture_logits'].cpu())
            all_gesture_labels.append(gesture_labels.cpu())
            all_skill_logits.append(outputs['skill_logits'].cpu())
            all_skill_labels.append(skill_labels.cpu())

            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Average losses
    avg_losses = {k: v / num_batches for k, v in all_losses.items()}

    # Concatenate all predictions
    all_kin_preds = torch.cat(all_kin_preds, dim=0)
    all_kin_targets = torch.cat(all_kin_targets, dim=0)
    all_gesture_logits = torch.cat(all_gesture_logits, dim=0)
    all_gesture_labels = torch.cat(all_gesture_labels, dim=0)
    all_skill_logits = torch.cat(all_skill_logits, dim=0)
    all_skill_labels = torch.cat(all_skill_labels, dim=0)

    # Compute metrics
    kin_metrics = compute_kinematics_metrics(all_kin_preds, all_kin_targets)
    gesture_metrics = compute_gesture_metrics(all_gesture_logits, all_gesture_labels)
    skill_metrics = compute_skill_metrics(all_skill_logits, all_skill_labels)

    # Combine all results
    results = {
        'losses': avg_losses,
        'kinematics': kin_metrics,
        'gesture': gesture_metrics,
        'skill': skill_metrics
    }

    return results


def print_results(results: Dict, mode: str = 'val'):
    """Pretty print evaluation results."""
    print("\n" + "=" * 60)
    print(f"Evaluation Results ({mode.upper()} set)")
    print("=" * 60)

    losses = results['losses']
    print("\nLosses:")
    print(f"  Total Loss: {losses.get('total', sum(losses.values())):.6f}")
    print(f"  Kinematics Loss: {losses.get('kin', 0):.6f}")
    if 'kin_pos' in losses:
        print(f"    - Position: {losses['kin_pos']:.6f}")
    if 'kin_rot' in losses:
        print(f"    - Rotation: {losses['kin_rot']:.6f} ({losses.get('kin_rot_deg', 0):.2f}°)")
    if 'kin_jaw' in losses:
        print(f"    - Jaw: {losses['kin_jaw']:.6f}")
    print(f"  Gesture Loss: {losses.get('gesture', 0):.6f}")
    print(f"  Skill Loss: {losses.get('skill', 0):.6f}")

    print("\nKinematics Metrics:")
    kin = results['kinematics']
    print(f"  Position RMSE: {kin['pos_rmse']:.6f}")
    print(f"  End-Effector Error: {kin['ee_error']:.6f}")
    print(f"  Rotation RMSE: {kin['rot_rmse']:.6f}")

    print("\nGesture Classification:")
    gesture = results['gesture']
    print(f"  Accuracy: {gesture['gesture_accuracy']*100:.2f}%")
    print(f"  F1 (Macro): {gesture['gesture_f1_macro']:.4f}")
    print(f"  F1 (Micro): {gesture['gesture_f1_micro']:.4f}")

    print("\nSkill Classification:")
    skill = results['skill']
    print(f"  Accuracy: {skill['skill_accuracy']*100:.2f}%")
    print(f"  F1 (Macro): {skill['skill_f1_macro']:.4f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Data root directory')
    parser.add_argument('--task', type=str, default='Knot_Tying',
                        help='Task name')
    parser.add_argument('--arm', type=str, default='PSM2',
                        help='Arm to use (PSM1 or PSM2)')
    parser.add_argument('--split', type=str, default=None,
                        help='Split name (e.g., fold_1). If None, uses all data.')
    parser.add_argument('--mode', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Which set to evaluate on')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')

    args = parser.parse_args()

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, config, epoch = load_model_from_checkpoint(args.checkpoint, device)
    print(f"Loaded model from epoch {epoch + 1}")

    # Load dataset
    print(f"\nLoading dataset (task: {args.task}, mode: {args.mode})...")

    # Create full dataset first
    full_dataset = JIGSAWSViTDataset(
        data_root=args.data_root,
        task=args.task,
        mode='train',  # Load all samples
        arm=args.arm
    )

    if args.split is not None:
        # Use split loader to get trial IDs for this split/mode
        split_loader = SplitLoader(args.data_root, args.task, args.split)

        if args.mode == 'train':
            trial_ids = split_loader.get_train_trials()
        elif args.mode == 'val':
            trial_ids = split_loader.get_val_trials()
        else:
            trial_ids = split_loader.get_test_trials()

        print(f"  Split: {args.split}, Mode: {args.mode}")
        print(f"  Trial IDs: {trial_ids}")

        # Filter dataset to only include samples from these trials
        indices = []
        for idx in range(len(full_dataset)):
            sample_trial = full_dataset.samples[idx].get('trial_id', '')
            # Check if any trial_id matches
            for tid in trial_ids:
                if tid in sample_trial or sample_trial in tid:
                    indices.append(idx)
                    break

        dataset = Subset(full_dataset, indices)
        print(f"  Filtered samples: {len(dataset)}")
    else:
        dataset = full_dataset
        print(f"  Using all {len(dataset)} samples (no split specified)")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Evaluate
    results = evaluate(model, dataloader, device, config)

    # Print results
    print_results(results, args.mode)

    # Return results for programmatic use
    return results


if __name__ == '__main__':
    main()
