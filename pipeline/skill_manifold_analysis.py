#!/usr/bin/env python3
"""Post-hoc analysis: does an existing checkpoint's embedding space separate
skill levels (Novice / Intermediate / Expert)?

For each sample in the fold's test set, we pool the model's temporal-aggregator
memory across time to a single 384-d vector, then measure:

  - within-skill distance: mean pairwise euclidean distance between samples
    of the same skill class
  - between-skill distance: mean pairwise euclidean distance between samples
    of different skill classes
  - separability ratio: between / within  (>1 -> classes are separable)
  - silhouette coefficient (sklearn) for the skill labeling

We also run the same analysis stratified by gesture class, so skill structure
isn't being confounded by gesture structure.

Outputs (under ``--output_dir``, default ``analysis/skill_manifold/``):
  - ``<stem>_metrics.json``: all numerical results
  - ``<stem>_pca.png``: 2D PCA scatter colored by skill level
  - ``<stem>_dispersion.png``: bar plot of within vs between distances

Usage (from repo root):
    python pipeline/skill_manifold_analysis.py \\
        --checkpoint checkpoints/brain_eye/all/fold_1/best_model.pth \\
        --data_root . \\
        --task all \\
        --split fold_1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from data.jigsaws_multitask_dataset import JIGSAWSMultiTaskDataset  # noqa: E402
from data.jigsaws_vit_dataset import JIGSAWSViTDataset  # noqa: E402
from data.split_loader import SplitLoader  # noqa: E402
from training.train_vit_system import (  # noqa: E402
    EEGInformedViTModel,
    filter_dataset_by_trials,
    pad_collate_fn,
)


SKILL_NAMES = ['Novice', 'Intermediate', 'Expert']  # index matches label: N=0, I=1, E=2


def _build_dataset(data_root: Path, task: str, split: str, split_family: str,
                   arm: str = 'PSM2', mode: str = 'test'):
    """Return (dataset, is_multitask) for the fold's {train,val,test} samples."""
    if task.lower() == 'all':
        ds = JIGSAWSMultiTaskDataset(
            data_root=str(data_root),
            split_name=split,
            mode=mode,
            arm=arm,
            split_family=split_family,
        )
        return ds, True

    full = JIGSAWSViTDataset(
        data_root=str(data_root),
        task=task,
        mode=mode,
        arm=arm,
    )
    sl = SplitLoader(str(data_root), task=task, split_name=split,
                     split_family=split_family)
    if mode == 'train':
        trials = sl.get_train_trials()
    elif mode == 'val':
        trials = sl.get_val_trials()
    else:
        trials = sl.get_test_trials()
    seg_filter = sl.get_segment_filter(mode)
    subset = filter_dataset_by_trials(full, trials, segment_filter=seg_filter)
    return subset, False


@torch.no_grad()
def _collect_embeddings(model, loader, device):
    """Run inference and return (embeddings Nx384, skill_labels N, gesture_labels N,
    task_labels N-or-None, trial_ids list[str])."""
    model.eval()
    model.to(device)

    embs, skills, gestures, tasks, trial_ids = [], [], [], [], []
    for batch in loader:
        rgb = batch['rgb'].to(device)
        kin = batch.get('kinematics')
        if kin is not None:
            kin = kin.to(device)
        out = model(rgb, target_kinematics=kin, teacher_forcing_prob=0.0,
                    return_embeddings=True)
        # memory: (B, T, 384) -> mean over T -> (B, 384)
        mem = out['memory']
        pooled = mem.mean(dim=1).cpu().numpy()
        embs.append(pooled)
        skills.append(batch['skill_label'].cpu().numpy())
        gestures.append(batch['gesture_label'].cpu().numpy())
        if 'task_label' in batch:
            tasks.append(batch['task_label'].cpu().numpy())
        if 'trial_id' in batch:
            tids = batch['trial_id']
            if isinstance(tids, (list, tuple)):
                trial_ids.extend([str(t) for t in tids])
            else:
                trial_ids.extend([str(tids)])

    embs = np.concatenate(embs, axis=0)
    skills = np.concatenate(skills, axis=0)
    gestures = np.concatenate(gestures, axis=0)
    tasks = np.concatenate(tasks, axis=0) if tasks else None
    return embs, skills, gestures, tasks, trial_ids


def _skill_separability(embs: np.ndarray, labels: np.ndarray):
    """Return dict with within-class / between-class mean distances and ratio.

    Uses pairwise euclidean. Class labels are integers. Silhouette included
    when sklearn is available.
    """
    unique = sorted(np.unique(labels).tolist())
    if len(unique) < 2 or embs.shape[0] < 4:
        return None

    # Pairwise distance matrix (NxN).
    diffs = embs[:, None, :] - embs[None, :, :]
    D = np.linalg.norm(diffs, axis=-1)

    within, between = {}, {}
    for c in unique:
        idx = labels == c
        if idx.sum() < 2:
            continue
        sub = D[np.ix_(idx, idx)]
        # Upper triangle excluding diagonal.
        iu, ju = np.triu_indices(sub.shape[0], k=1)
        within[int(c)] = float(sub[iu, ju].mean()) if len(iu) else float('nan')

    for i, c1 in enumerate(unique):
        for c2 in unique[i + 1:]:
            m1 = labels == c1
            m2 = labels == c2
            if m1.sum() == 0 or m2.sum() == 0:
                continue
            sub = D[np.ix_(m1, m2)]
            between[f'{int(c1)}_{int(c2)}'] = float(sub.mean())

    # Aggregate ratio: mean within vs mean between (harmonic, comparable).
    w_mean = float(np.mean(list(within.values()))) if within else float('nan')
    b_mean = float(np.mean(list(between.values()))) if between else float('nan')
    ratio = b_mean / w_mean if w_mean > 0 else float('nan')

    result = {
        'within_class_distance': within,
        'between_class_distance': between,
        'within_mean': w_mean,
        'between_mean': b_mean,
        'separability_ratio': ratio,
        'n_samples': int(embs.shape[0]),
        'class_counts': {int(c): int((labels == c).sum()) for c in unique},
    }

    try:
        from sklearn.metrics import silhouette_score
        if len(unique) >= 2 and all(v >= 2 for v in result['class_counts'].values()):
            result['silhouette'] = float(silhouette_score(embs, labels, metric='euclidean'))
    except ImportError:
        pass

    return result


def _pca_scatter(embs, skills, gestures, out_path: Path, title: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    xy = pca.fit_transform(embs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    colors = ['#d73027', '#fee090', '#1a9850']  # N, I, E
    for c in [0, 1, 2]:
        mask = skills == c
        if mask.sum() == 0:
            continue
        axes[0].scatter(xy[mask, 0], xy[mask, 1], s=18, alpha=0.7,
                        c=colors[c], label=SKILL_NAMES[c], edgecolors='none')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
    axes[0].set_title(f'{title} — colored by skill')
    axes[0].legend(loc='best', fontsize=8)

    # gesture-colored scatter for contrast
    sc = axes[1].scatter(xy[:, 0], xy[:, 1], s=18, alpha=0.7,
                         c=gestures, cmap='tab20', edgecolors='none')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
    axes[1].set_title(f'{title} — colored by gesture (for contrast)')
    plt.colorbar(sc, ax=axes[1], fraction=0.046, label='gesture idx')

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def _dispersion_plot(metrics: dict, out_path: Path, title: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    within = metrics['within_class_distance']
    between = metrics['between_class_distance']

    bars = []
    for c, v in sorted(within.items()):
        bars.append((f'within {SKILL_NAMES[c]}', v, '#1f77b4'))
    for pair, v in sorted(between.items()):
        c1, c2 = pair.split('_')
        bars.append((f'between {SKILL_NAMES[int(c1)]}-{SKILL_NAMES[int(c2)]}',
                     v, '#d62728'))

    xs = np.arange(len(bars))
    heights = [b[1] for b in bars]
    colors = [b[2] for b in bars]
    labels = [b[0] for b in bars]

    ax.bar(xs, heights, color=colors, edgecolor='white')
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Mean pairwise euclidean distance')
    ax.set_title(f'{title}\nseparability = between / within = '
                 f'{metrics["separability_ratio"]:.3f}')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def analyze(
    checkpoint: Path,
    data_root: Path,
    task: str,
    split: str,
    split_family: str,
    output_dir: Path,
    device: str = 'cuda',
    batch_size: int = 8,
    arm: str = 'PSM2',
    stem: str = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or checkpoint.parent.parent.name + '_' + checkpoint.parent.name

    print(f'\n[skill_manifold] checkpoint: {checkpoint}')
    print(f'[skill_manifold] split:      {split} ({split_family})')
    ckpt = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    config = ckpt.get('config') or {}
    if not config:
        # Some older checkpoints may not embed config. Fall back to a minimal default.
        config = {'brain_mode': 'none', 'model_name': 'vit_small_patch16_224'}

    # We are loading a checkpoint, so pretrained ViT weights will be overwritten.
    # Skip the hub download to avoid needing internet at analysis time.
    config = dict(config)
    config['pretrained'] = False

    model = EEGInformedViTModel(config)
    state = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  warn: {len(missing)} missing keys (e.g. {missing[:3]})')
    if unexpected:
        print(f'  warn: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})')

    dataset, is_multi = _build_dataset(data_root, task, split, split_family, arm)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=pad_collate_fn,
    )
    print(f'[skill_manifold] test samples: {len(dataset)}')

    dev = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
    embs, skills, gestures, tasks, trial_ids = _collect_embeddings(model, loader, dev)
    print(f'[skill_manifold] embeddings shape: {embs.shape}')

    metrics_overall = _skill_separability(embs, skills)
    per_gesture = {}
    for g in np.unique(gestures):
        mask = gestures == g
        if mask.sum() < 4:
            continue
        m = _skill_separability(embs[mask], skills[mask])
        if m is not None:
            per_gesture[int(g)] = m

    out = {
        'checkpoint': str(checkpoint),
        'split': split,
        'split_family': split_family,
        'task': task,
        'n_samples': int(embs.shape[0]),
        'overall': metrics_overall,
        'per_gesture': per_gesture,
        'config_brain_mode': config.get('brain_mode'),
    }
    json_path = output_dir / f'{stem}_metrics.json'
    json_path.write_text(json.dumps(out, indent=2))
    print(f'[skill_manifold] wrote {json_path}')

    if metrics_overall is not None:
        _dispersion_plot(metrics_overall, output_dir / f'{stem}_dispersion.png',
                         title=stem)
        _pca_scatter(embs, skills, gestures, output_dir / f'{stem}_pca.png',
                     title=stem)
        print(f'[skill_manifold] wrote {output_dir / f"{stem}_dispersion.png"}')
        print(f'[skill_manifold] wrote {output_dir / f"{stem}_pca.png"}')

    return out


@torch.no_grad()
def _infer_fold(checkpoint: Path, data_root: Path, task: str, split: str,
                split_family: str, device: str, batch_size: int, arm: str,
                mode: str = 'test'):
    """Load checkpoint, run inference on the fold's {train,val,test} subset,
    return raw numpy arrays."""
    ckpt = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    config = dict(ckpt.get('config') or {})
    config['pretrained'] = False
    config.setdefault('brain_mode', 'none')
    config.setdefault('model_name', 'vit_small_patch16_224')

    model = EEGInformedViTModel(config)
    state = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
    model.load_state_dict(state, strict=False)

    dataset, _ = _build_dataset(data_root, task, split, split_family, arm, mode=mode)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=pad_collate_fn,
    )
    dev = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
    return _collect_embeddings(model, loader, dev)


def analyze_condition_aggregate(
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
    """Run inference per fold using its own checkpoint, concatenate the
    per-fold test-set embeddings, and compute skill-separability on the union.

    Honest cross-validated embedding analysis: each point is predicted by a
    model that never trained on that point. Across all folds we typically get
    all skill levels represented.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_ckpts = sorted(aggregate_root.glob('fold_*/best_model.pth'))
    if not fold_ckpts:
        raise FileNotFoundError(f'No fold_*/best_model.pth under {aggregate_root}')

    all_embs, all_skills, all_gestures, all_folds = [], [], [], []
    per_fold_records = []
    for ck in fold_ckpts:
        fold_name = ck.parent.name  # e.g. "fold_1"
        print(f'\n[aggregate] {fold_name}: {ck}')
        try:
            embs, skills, gestures, tasks, trial_ids = _infer_fold(
                ck, data_root, task, fold_name, split_family,
                device, batch_size, arm,
            )
        except Exception as e:
            print(f'  skip ({type(e).__name__}: {e})')
            continue
        all_embs.append(embs)
        all_skills.append(skills)
        all_gestures.append(gestures)
        all_folds.append(np.full(len(embs), int(fold_name.split('_')[1])))
        per_fold_records.append({
            'fold': fold_name,
            'n_samples': int(len(embs)),
            'skills_present': sorted(set(int(s) for s in skills.tolist())),
        })

    if not all_embs:
        raise RuntimeError('No folds produced embeddings.')
    embs = np.concatenate(all_embs, axis=0)
    skills = np.concatenate(all_skills, axis=0)
    gestures = np.concatenate(all_gestures, axis=0)
    folds = np.concatenate(all_folds, axis=0)

    metrics_overall = _skill_separability(embs, skills)
    per_gesture = {}
    for g in np.unique(gestures):
        mask = gestures == g
        if mask.sum() < 4:
            continue
        m = _skill_separability(embs[mask], skills[mask])
        if m is not None:
            per_gesture[int(g)] = m

    out = {
        'aggregate_root': str(aggregate_root),
        'fold_checkpoints': [str(p) for p in fold_ckpts],
        'task': task,
        'split_family': split_family,
        'per_fold': per_fold_records,
        'n_samples_total': int(embs.shape[0]),
        'overall': metrics_overall,
        'per_gesture': per_gesture,
    }
    json_path = output_dir / f'{stem}_aggregate_metrics.json'
    json_path.write_text(json.dumps(out, indent=2))
    print(f'\n[aggregate] wrote {json_path}')

    if metrics_overall is not None:
        _dispersion_plot(metrics_overall,
                         output_dir / f'{stem}_aggregate_dispersion.png',
                         title=f'{stem} (aggregate across folds)')
        _pca_scatter(embs, skills, gestures,
                     output_dir / f'{stem}_aggregate_pca.png',
                     title=f'{stem} (aggregate)')
        print(f'[aggregate] wrote {output_dir / f"{stem}_aggregate_dispersion.png"}')
        print(f'[aggregate] wrote {output_dir / f"{stem}_aggregate_pca.png"}')

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path,
                   help='Single checkpoint (requires --split). Mutually exclusive with --aggregate_root.')
    p.add_argument('--aggregate_root', type=Path,
                   help='Directory containing fold_*/best_model.pth. Runs each fold '
                        'against its own split and aggregates embeddings.')
    p.add_argument('--data_root', type=Path, default=Path('.'))
    p.add_argument('--task', type=str, default='all',
                   help='Task name, or "all" for multi-task checkpoints')
    p.add_argument('--split', type=str, default=None,
                   help='Required when using --checkpoint; ignored when using --aggregate_root.')
    p.add_argument('--split_family', type=str, default='louo',
                   choices=['louo', 'inter_trial_within_subject', 'intra_trial_half'])
    p.add_argument('--output_dir', type=Path, default=Path('analysis/skill_manifold'))
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--arm', type=str, default='PSM2')
    p.add_argument('--stem', type=str, default=None,
                   help='Filename stem for outputs.')
    args = p.parse_args()

    if args.aggregate_root:
        stem = args.stem or args.aggregate_root.name
        analyze_condition_aggregate(
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
    elif args.checkpoint:
        if not args.split:
            raise SystemExit('--split is required with --checkpoint')
        analyze(
            checkpoint=args.checkpoint.resolve(),
            data_root=args.data_root.resolve(),
            task=args.task,
            split=args.split,
            split_family=args.split_family,
            output_dir=args.output_dir.resolve(),
            device=args.device,
            batch_size=args.batch_size,
            arm=args.arm,
            stem=args.stem,
        )
    else:
        raise SystemExit('Provide either --checkpoint or --aggregate_root.')


if __name__ == '__main__':
    main()
