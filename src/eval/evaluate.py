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
import re
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
from eval.metrics import (
    compute_kinematics_metrics,
    compute_gesture_metrics,
    compute_skill_metrics,
    compute_gesture_edit_distance,
    compute_frame_weighted_gesture_accuracy,
    compute_gesture_frame_iou,
    compute_ordinal_skill_metrics,
)
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
    all_trial_ids: List[str] = []
    all_start_frames: List[int] = []
    all_end_frames: List[int] = []

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

            # Per-segment identity + frame range (needed for sequence / IoU metrics)
            tids = batch.get('trial_id')
            if isinstance(tids, (list, tuple)):
                all_trial_ids.extend(str(t) for t in tids)
            sf = batch.get('start_frame')
            ef = batch.get('end_frame')
            if isinstance(sf, torch.Tensor):
                all_start_frames.extend(sf.tolist())
            if isinstance(ef, torch.Tensor):
                all_end_frames.extend(ef.tolist())

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

    # Ordinal skill metrics (N < I < E ordering)
    skill_metrics.update(compute_ordinal_skill_metrics(all_skill_logits, all_skill_labels))

    # Sequence-level gesture metrics (require per-segment trial_id + frame range)
    N = all_gesture_logits.shape[0]
    if len(all_trial_ids) == N and len(all_start_frames) == N and len(all_end_frames) == N:
        pred_per_seg = all_gesture_logits.argmax(dim=-1).tolist()
        true_per_seg = all_gesture_labels.tolist()
        durations = [max(0, e - s) for s, e in zip(all_start_frames, all_end_frames)]

        # Group segments by trial_id, sort by start_frame, build per-trial sequences
        trial_bins: Dict[str, List[tuple]] = {}
        for tid, sfr, p, t in zip(all_trial_ids, all_start_frames, pred_per_seg, true_per_seg):
            trial_bins.setdefault(tid, []).append((sfr, p, t))
        pred_seqs, true_seqs = [], []
        for tid in sorted(trial_bins.keys()):
            items = sorted(trial_bins[tid], key=lambda x: x[0])
            pred_seqs.append([p for _, p, _ in items])
            true_seqs.append([t for _, _, t in items])

        gesture_metrics.update(compute_gesture_edit_distance(pred_seqs, true_seqs))
        gesture_metrics['gesture_frame_weighted_accuracy'] = (
            compute_frame_weighted_gesture_accuracy(pred_per_seg, true_per_seg, durations)
        )
        gesture_metrics.update(
            compute_gesture_frame_iou(pred_per_seg, true_per_seg, durations, num_classes=15)
        )

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
    if 'gesture_frame_weighted_accuracy' in gesture:
        print(f"  Frame-weighted Accuracy: {gesture['gesture_frame_weighted_accuracy']*100:.2f}%")
    if 'edit_distance_mean' in gesture:
        print(f"  Edit Distance: {gesture['edit_distance_mean']:.3f} "
              f"(normalized: {gesture.get('edit_distance_normalized', 0):.3f})")
    if 'iou_mean' in gesture:
        print(f"  Frame IoU (mean over {gesture['iou_classes_seen']} classes): "
              f"{gesture['iou_mean']:.4f}")

    print("\nSkill Classification:")
    skill = results['skill']
    print(f"  Accuracy: {skill['skill_accuracy']*100:.2f}%")
    print(f"  F1 (Macro): {skill['skill_f1_macro']:.4f}")
    if 'ord_mae' in skill:
        print(f"  Ordinal MAE (argmax): {skill['ord_mae']:.4f}")
        print(f"  Ordinal MAE (expected): {skill.get('ord_expected_mae', 0):.4f}")
        print(f"  Ordinal Spearman ρ: {skill.get('ord_spearman', 0):.4f}")

    print("=" * 60)


def save_aggregate_compatible_report(
    results: Dict,
    output_dir: Path,
    task: str,
    split: Optional[str],
    mode: str,
    epoch: int,
) -> Optional[Path]:
    """
    Write metrics in sections understood by aggregate_louo_results.py
    (Loss Components, Kinematics Metrics, Gesture Metrics, Skill Metrics).
    """
    m = re.match(r"fold_(\d+)", split or "")
    if not m:
        return None
    fold_num = int(m.group(1))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{task}_test_fold_{fold_num}_results.txt"

    lines = []
    lines.append("=" * 60)
    lines.append(f"Evaluation Results: {task}")
    lines.append(f"Mode: {mode.upper()}")
    lines.append(f"Split: {split}")
    lines.append(f"Checkpoint Epoch: {epoch + 1}")
    lines.append("=" * 60)

    losses = results["losses"]
    lines.append("")
    lines.append("Loss Components")
    lines.append(f"  Total Loss: {losses.get('total', sum(losses.values())):.6f}")
    lines.append(f"  Kinematics Loss: {losses.get('kin', 0):.6f}")
    if "kin_pos" in losses:
        lines.append(f"  Position: {losses['kin_pos']:.6f}")
    if "kin_rot" in losses:
        lines.append(f"  Rotation: {losses['kin_rot']:.6f}")
    if "kin_jaw" in losses:
        lines.append(f"  Jaw: {losses['kin_jaw']:.6f}")
    lines.append(f"  Gesture Loss: {losses.get('gesture', 0):.6f}")
    lines.append(f"  Skill Loss: {losses.get('skill', 0):.6f}")

    kin = results["kinematics"]
    lines.append("")
    lines.append("Kinematics Metrics")
    lines.append(f"  Position RMSE: {kin['pos_rmse']:.6f}")
    lines.append(f"  End-Effector Error: {kin['ee_error']:.6f}")
    lines.append(f"  Rotation RMSE: {kin['rot_rmse']:.6f}")

    gesture = results["gesture"]
    lines.append("")
    lines.append("Gesture Metrics")
    lines.append(f"  Accuracy: {gesture['gesture_accuracy'] * 100:.2f}%")
    lines.append(f"  F1 Macro: {gesture['gesture_f1_macro']:.4f}")
    lines.append(f"  F1 Micro: {gesture['gesture_f1_micro']:.4f}")
    if 'gesture_frame_weighted_accuracy' in gesture:
        lines.append(f"  Frame-weighted Accuracy: {gesture['gesture_frame_weighted_accuracy'] * 100:.2f}%")
    if 'edit_distance_mean' in gesture:
        lines.append(f"  Edit Distance: {gesture['edit_distance_mean']:.3f}")
        lines.append(f"  Edit Distance Normalized: {gesture.get('edit_distance_normalized', 0):.3f}")
    if 'iou_mean' in gesture:
        lines.append(f"  Frame IoU Mean: {gesture['iou_mean']:.4f}")

    skill = results["skill"]
    lines.append("")
    lines.append("Skill Metrics")
    lines.append(f"  Accuracy: {skill['skill_accuracy'] * 100:.2f}%")
    lines.append(f"  F1 Macro: {skill['skill_f1_macro']:.4f}")
    if 'ord_mae' in skill:
        lines.append(f"  Ordinal MAE: {skill['ord_mae']:.4f}")
        lines.append(f"  Ordinal Expected MAE: {skill.get('ord_expected_mae', 0):.4f}")
        lines.append(f"  Ordinal Spearman: {skill.get('ord_spearman', 0):.4f}")
    lines.append("=" * 60)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


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
    parser.add_argument('--output_dir', type=str, default=None,
                        help='If set, write a report file for aggregate_louo_results.py '
                             '(requires --split fold_N).')

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

    if args.output_dir and args.split:
        out = save_aggregate_compatible_report(
            results,
            Path(args.output_dir),
            args.task,
            args.split,
            args.mode,
            epoch,
        )
        if out:
            print(f"\nWrote aggregate-compatible report: {out}")

    # Return results for programmatic use
    return results


if __name__ == '__main__':
    main()
