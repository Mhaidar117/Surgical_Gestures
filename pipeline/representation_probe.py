#!/usr/bin/env python3
"""Post-hoc linear + k-NN probes on a condition's cross-fold embeddings.

Answers: **what does the representation actually encode?**

Per condition, aggregates per-sample pooled embeddings across folds (each fold
uses its own held-out checkpoint — no leakage), then for each probe target
(skill / gesture / task / surgeon):

  - trains a ridge-regularized logistic regression with 5-fold CV
  - trains a k-NN classifier (k=5) with 5-fold CV
  - reports accuracy, balanced accuracy, and a chance baseline

Surgeon-ID probe: **lower is better**. A strong surgeon probe means the model
has memorized motor style; a surgeon-invariant representation should probe
near chance.

Usage:
    python pipeline/representation_probe.py \\
        --aggregate_root checkpoints/brain_eye/all \\
        --data_root . --task all --split_family louo \\
        --output_dir analysis/representation_probe \\
        --stem brain_eye
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'pipeline'))

# Reuse the aggregate-across-folds inference helper from the skill manifold script.
from skill_manifold_analysis import _infer_fold  # type: ignore


SKILL_NAMES = ['Novice', 'Intermediate', 'Expert']


def _probe_target(
    embs: np.ndarray,
    labels: np.ndarray,
    target_name: str,
    n_folds: int = 5,
) -> Dict[str, float]:
    """Run ridge logistic regression + k-NN probe under 5-fold CV.

    Returns per-probe accuracy, balanced accuracy, and the majority-class
    baseline. If only one class is present, returns baseline-only dict.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    from collections import Counter

    classes = np.unique(labels)
    if classes.size < 2:
        return {
            'target': target_name,
            'n_classes_present': int(classes.size),
            'warning': 'only one class present; probe undefined',
        }

    counts = Counter(labels.tolist())
    majority_count = max(counts.values())
    baseline = majority_count / len(labels)
    min_class = min(counts.values())
    if min_class < n_folds:
        n_folds = max(2, min_class)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    lr_accs, lr_bals = [], []
    knn_accs, knn_bals = [], []
    for train_idx, test_idx in skf.split(embs, labels):
        scaler = StandardScaler().fit(embs[train_idx])
        Xtr = scaler.transform(embs[train_idx])
        Xte = scaler.transform(embs[test_idx])
        ytr, yte = labels[train_idx], labels[test_idx]

        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(Xtr, ytr)
        lr_pred = lr.predict(Xte)
        lr_accs.append(accuracy_score(yte, lr_pred))
        lr_bals.append(balanced_accuracy_score(yte, lr_pred))

        knn = KNeighborsClassifier(n_neighbors=min(5, len(ytr) - 1))
        knn.fit(Xtr, ytr)
        knn_pred = knn.predict(Xte)
        knn_accs.append(accuracy_score(yte, knn_pred))
        knn_bals.append(balanced_accuracy_score(yte, knn_pred))

    return {
        'target': target_name,
        'n_classes_present': int(classes.size),
        'n_samples': int(len(labels)),
        'majority_baseline': float(baseline),
        'linear_accuracy_mean': float(np.mean(lr_accs)),
        'linear_accuracy_std': float(np.std(lr_accs)),
        'linear_balanced_mean': float(np.mean(lr_bals)),
        'knn_accuracy_mean': float(np.mean(knn_accs)),
        'knn_accuracy_std': float(np.std(knn_accs)),
        'knn_balanced_mean': float(np.mean(knn_bals)),
    }


def run(
    aggregate_root: Path,
    data_root: Path,
    task: str,
    split_family: str,
    output_dir: Path,
    stem: str,
    device: str = 'cuda',
    batch_size: int = 8,
    arm: str = 'PSM2',
):
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_ckpts = sorted(aggregate_root.glob('fold_*/best_model.pth'))
    if not fold_ckpts:
        raise FileNotFoundError(f'No fold_*/best_model.pth under {aggregate_root}')

    all_embs: List[np.ndarray] = []
    all_skills: List[np.ndarray] = []
    all_gestures: List[np.ndarray] = []
    all_tasks: List[np.ndarray] = []
    all_surgeons: List[np.ndarray] = []
    per_fold_records: List[dict] = []

    for ck in fold_ckpts:
        fold_name = ck.parent.name
        print(f'\n[probe] {fold_name}: {ck}')
        try:
            embs, skills, gestures, tasks, trial_ids = _infer_fold(
                ck, data_root, task, fold_name, split_family,
                device, batch_size, arm,
            )
        except Exception as e:
            print(f'  skip ({type(e).__name__}: {e})')
            continue
        # Extract surgeon letter from trial_id (e.g. 'Knot_Tying_B001' -> 'B').
        surgeon_letters = []
        for tid in trial_ids:
            if isinstance(tid, str) and '_' in tid:
                suffix = tid.rsplit('_', 1)[-1]
                surgeon_letters.append(suffix[0] if suffix else '?')
            else:
                surgeon_letters.append('?')
        surgeon_letters = np.array(surgeon_letters)

        all_embs.append(embs)
        all_skills.append(skills)
        all_gestures.append(gestures)
        if tasks is not None:
            all_tasks.append(tasks)
        all_surgeons.append(surgeon_letters)
        per_fold_records.append({
            'fold': fold_name,
            'n_samples': int(len(embs)),
            'surgeons_present': sorted(set(surgeon_letters.tolist())),
        })

    if not all_embs:
        raise RuntimeError('No folds produced embeddings.')

    embs = np.concatenate(all_embs, axis=0)
    skills = np.concatenate(all_skills, axis=0).astype(np.int64)
    gestures = np.concatenate(all_gestures, axis=0).astype(np.int64)
    tasks = np.concatenate(all_tasks, axis=0).astype(np.int64) if all_tasks else None
    surgeons_letters = np.concatenate(all_surgeons, axis=0)
    # Map surgeon letter -> int label
    letter_to_id = {s: i for i, s in enumerate(sorted(set(surgeons_letters.tolist())))}
    surgeons = np.array([letter_to_id[s] for s in surgeons_letters], dtype=np.int64)

    print(f'\n[probe] aggregated {embs.shape[0]} samples, embedding dim={embs.shape[1]}')
    print(f'[probe] surgeons present: {sorted(letter_to_id)}')

    probes = {
        'skill': _probe_target(embs, skills, 'skill'),
        'gesture': _probe_target(embs, gestures, 'gesture'),
        'surgeon': _probe_target(embs, surgeons, 'surgeon'),
    }
    if tasks is not None:
        probes['task'] = _probe_target(embs, tasks, 'task')

    out = {
        'aggregate_root': str(aggregate_root),
        'fold_checkpoints': [str(p) for p in fold_ckpts],
        'per_fold': per_fold_records,
        'n_samples_total': int(embs.shape[0]),
        'embedding_dim': int(embs.shape[1]),
        'surgeon_letter_to_id': letter_to_id,
        'probes': probes,
    }
    json_path = output_dir / f'{stem}_probe.json'
    json_path.write_text(json.dumps(out, indent=2))
    print(f'\n[probe] wrote {json_path}')

    # Print compact summary table.
    print('\n' + '=' * 70)
    print(f'{stem} — representation probe (5-fold CV)')
    print('=' * 70)
    print(f'{"target":<10} {"n_cls":>6} {"baseline":>10} {"linear":>16} {"knn":>16}')
    for name, p in probes.items():
        if 'linear_accuracy_mean' not in p:
            print(f'{name:<10} {p.get("n_classes_present", 0):>6}  '
                  f'{p.get("warning", "")}')
            continue
        la = f'{p["linear_accuracy_mean"]:.3f}±{p["linear_accuracy_std"]:.3f}'
        ka = f'{p["knn_accuracy_mean"]:.3f}±{p["knn_accuracy_std"]:.3f}'
        print(f'{name:<10} {p["n_classes_present"]:>6} {p["majority_baseline"]:>10.3f} '
              f'{la:>16} {ka:>16}')
    print('=' * 70)
    print('Note: surgeon probe — LOWER is better (surgeon-invariant features).')
    return out


def run_single_checkpoint(
    checkpoint: Path,
    data_root: Path,
    task: str,
    split: str,
    split_family: str,
    mode: str,
    output_dir: Path,
    stem: str,
    device: str = 'cuda',
    batch_size: int = 8,
    arm: str = 'PSM2',
):
    """Probe one checkpoint on one split's {train,val,test} subset.

    When ``mode='train'`` on a LOUO fold, the probe sees all training surgeons
    (6 of 8), so skill/surgeon probes are well-defined. Useful for asking
    "what does THIS checkpoint's embedding space encode?" without needing
    cross-fold aggregation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'\n[probe-single] checkpoint: {checkpoint}')
    print(f'[probe-single] split: {split} ({split_family}), probe mode: {mode}')
    embs, skills, gestures, tasks, trial_ids = _infer_fold(
        checkpoint, data_root, task, split, split_family, device, batch_size, arm, mode=mode,
    )

    surgeon_letters = []
    for tid in trial_ids:
        if isinstance(tid, str) and '_' in tid:
            suffix = tid.rsplit('_', 1)[-1]
            surgeon_letters.append(suffix[0] if suffix else '?')
        else:
            surgeon_letters.append('?')
    surgeon_letters = np.array(surgeon_letters)
    letter_to_id = {s: i for i, s in enumerate(sorted(set(surgeon_letters.tolist())))}
    surgeons = np.array([letter_to_id[s] for s in surgeon_letters], dtype=np.int64)
    skills = skills.astype(np.int64)
    gestures = gestures.astype(np.int64)
    tasks = tasks.astype(np.int64) if tasks is not None else None

    print(f'[probe-single] {embs.shape[0]} samples, embedding dim={embs.shape[1]}')
    print(f'[probe-single] surgeons: {sorted(letter_to_id)}  skills: {sorted(set(skills.tolist()))}')

    probes = {
        'skill': _probe_target(embs, skills, 'skill'),
        'gesture': _probe_target(embs, gestures, 'gesture'),
        'surgeon': _probe_target(embs, surgeons, 'surgeon'),
    }
    if tasks is not None:
        probes['task'] = _probe_target(embs, tasks, 'task')

    out = {
        'checkpoint': str(checkpoint),
        'split': split,
        'split_family': split_family,
        'probe_mode': mode,
        'n_samples': int(embs.shape[0]),
        'embedding_dim': int(embs.shape[1]),
        'surgeon_letter_to_id': letter_to_id,
        'probes': probes,
    }
    json_path = output_dir / f'{stem}_probe.json'
    json_path.write_text(json.dumps(out, indent=2))
    print(f'[probe-single] wrote {json_path}')

    print('\n' + '=' * 70)
    print(f'{stem} — single-checkpoint probe ({mode}-set, 5-fold CV)')
    print('=' * 70)
    print(f'{"target":<10} {"n_cls":>6} {"baseline":>10} {"linear":>16} {"knn":>16}')
    for name, pr in probes.items():
        if 'linear_accuracy_mean' not in pr:
            print(f'{name:<10} {pr.get("n_classes_present", 0):>6}  '
                  f'{pr.get("warning", "")}')
            continue
        la = f'{pr["linear_accuracy_mean"]:.3f}±{pr["linear_accuracy_std"]:.3f}'
        ka = f'{pr["knn_accuracy_mean"]:.3f}±{pr["knn_accuracy_std"]:.3f}'
        print(f'{name:<10} {pr["n_classes_present"]:>6} {pr["majority_baseline"]:>10.3f} '
              f'{la:>16} {ka:>16}')
    print('=' * 70)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode_group = p.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--aggregate_root', type=Path,
                            help='Directory containing fold_*/best_model.pth')
    mode_group.add_argument('--checkpoint', type=Path,
                            help='Single checkpoint path (requires --split).')
    p.add_argument('--data_root', type=Path, default=Path('.'))
    p.add_argument('--task', type=str, default='all')
    p.add_argument('--split', type=str, default=None,
                   help='Required with --checkpoint.')
    p.add_argument('--split_family', type=str, default='louo',
                   choices=['louo', 'inter_trial_within_subject', 'intra_trial_half'])
    p.add_argument('--probe_mode', type=str, default='train',
                   choices=['train', 'val', 'test'],
                   help='Which subset of the fold to probe (single-checkpoint mode only). '
                        'train is the default because it spans diverse surgeons/skills, '
                        'letting surgeon/skill probes function.')
    p.add_argument('--output_dir', type=Path, default=Path('analysis/representation_probe'))
    p.add_argument('--stem', type=str, default=None)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--arm', type=str, default='PSM2')
    args = p.parse_args()

    if args.aggregate_root:
        stem = args.stem or args.aggregate_root.name
        run(
            aggregate_root=args.aggregate_root.resolve(),
            data_root=args.data_root.resolve(),
            task=args.task,
            split_family=args.split_family,
            output_dir=args.output_dir.resolve(),
            stem=stem,
            device=args.device,
            batch_size=args.batch_size,
            arm=args.arm,
        )
    else:
        if not args.split:
            raise SystemExit('--split is required with --checkpoint')
        stem = args.stem or f'{args.checkpoint.parent.name}_{args.split}'
        run_single_checkpoint(
            checkpoint=args.checkpoint.resolve(),
            data_root=args.data_root.resolve(),
            task=args.task,
            split=args.split,
            split_family=args.split_family,
            mode=args.probe_mode,
            output_dir=args.output_dir.resolve(),
            stem=stem,
            device=args.device,
            batch_size=args.batch_size,
            arm=args.arm,
        )


if __name__ == '__main__':
    main()
