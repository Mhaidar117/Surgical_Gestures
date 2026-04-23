#!/usr/bin/env python3
"""Hyperparameter sweep driver for train_vit_system.py.

Generates the Cartesian product of override values over a base YAML config,
launches each run sequentially as a subprocess, then aggregates each run's
``best_model.pth`` val metrics into a ranked summary.

Usage example (brain-weight sweep for kinematics_rsa on fold_1):

    python pipeline/run_hparam_sweep.py \\
        --base_config src/configs/kinematics_rsa.yaml \\
        --sweep '{"loss_weights.brain": [0.01, 0.05, 0.1, 0.2]}' \\
        --task all --split fold_1 --split_family louo \\
        --output_root checkpoints/sweeps/kin_rsa_brain_weight \\
        --primary_metric gesture

For multi-axis sweeps, add keys:

    --sweep '{"loss_weights.brain": [0.05, 0.1], "loss_weights.skill_contra": [0, 0.1]}'

Dry-run prints the planned runs without launching anything:

    ... --dry_run
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dataset_paths import resolve_dataset_root


def _deep_set(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested config value using a dotted path like 'loss_weights.brain'."""
    parts = dotted_key.split('.')
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _run_name(axes: Dict[str, Any]) -> str:
    """Compact filesystem-safe name from axes dict."""
    parts = []
    for k, v in axes.items():
        short = k.replace('.', '_').replace('loss_weights', 'w')
        parts.append(f'{short}_{v}')
    return '__'.join(parts).replace(' ', '')


def _extract_best_metrics(ckpt_path: Path) -> Dict[str, float]:
    """Load val_losses dict from a best_model.pth."""
    if not ckpt_path.exists():
        return {}
    try:
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    except Exception as e:
        print(f'  [warn] could not read {ckpt_path}: {e}')
        return {}
    vl = ckpt.get('val_losses')
    if not isinstance(vl, dict):
        return {}
    # Coerce tensors to floats.
    out = {}
    for k, v in vl.items():
        if hasattr(v, 'item'):
            try:
                out[k] = float(v.item())
            except Exception:
                pass
        elif isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def _launch_run(
    base_cfg: Dict[str, Any],
    axes: Dict[str, Any],
    output_root: Path,
    args: argparse.Namespace,
) -> Tuple[str, Path, int, Dict[str, float]]:
    """Materialize a config with overrides, launch train_vit_system, collect metrics.

    Returns (run_name, output_dir, returncode, metrics).
    """
    cfg = copy.deepcopy(base_cfg)
    for k, v in axes.items():
        _deep_set(cfg, k, v)
    run_name = _run_name(axes)
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / 'config.yaml'
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    cmd = [
        sys.executable,
        str(REPO / 'src' / 'training' / 'train_vit_system.py'),
        '--config', str(cfg_path),
        '--data_root', str(Path(args.data_root)),
        '--task', args.task,
        '--split', args.split,
        '--split_family', args.split_family,
        '--output_dir', str(run_dir),
        '--arm', args.arm,
    ]
    log_path = run_dir / 'train.log'
    print(f'\n[sweep] launching {run_name} -> {run_dir}')
    print(f'  axes: {axes}')

    env = os.environ.copy()
    env['PYTHONPATH'] = str(REPO / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    # Some conda envs set SSL_CERT_FILE to a non-existent path, which breaks
    # Hugging Face downloads inside timm during model init. Fall back to
    # certifi's bundled cert if the env-configured one is missing.
    cert = env.get('SSL_CERT_FILE')
    if not cert or not Path(cert).exists():
        try:
            import certifi  # type: ignore
            env['SSL_CERT_FILE'] = certifi.where()
        except ImportError:
            env.pop('SSL_CERT_FILE', None)

    if args.dry_run:
        print(f'  [dry_run] would execute: {" ".join(cmd)}')
        return run_name, run_dir, 0, {}

    with open(log_path, 'w', encoding='utf-8') as logf:
        proc = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=logf,
                              stderr=subprocess.STDOUT)
    metrics = _extract_best_metrics(run_dir / 'best_model.pth')
    return run_name, run_dir, proc.returncode, metrics


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base_config', type=Path, required=True)
    p.add_argument('--sweep', type=str, required=True,
                   help='JSON dict of dotted-key -> list-of-values, e.g. '
                        '\'{"loss_weights.brain": [0.01, 0.05]}\'')
    p.add_argument(
        '--data_root',
        type=str,
        default=None,
        help='Dataset root (default: env / iCloud path / repo).',
    )
    p.add_argument('--task', type=str, default='all')
    p.add_argument('--split', type=str, default='fold_1')
    p.add_argument('--split_family', type=str, default='louo',
                   choices=['louo', 'inter_trial_within_subject', 'intra_trial_half'])
    p.add_argument('--arm', type=str, default='PSM2')
    p.add_argument('--output_root', type=Path, required=True)
    p.add_argument('--primary_metric', type=str, default='gesture',
                   help='Key from val_losses to rank by; lower is better. '
                        'Common: total, gesture, skill, kin, brain_rsa.')
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()
    args.data_root = str(
        resolve_dataset_root(args.data_root, fallback_repo_root=REPO)
    )

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)
    sweep = json.loads(args.sweep)
    if not sweep:
        raise SystemExit('--sweep must be non-empty')

    axis_keys = list(sweep.keys())
    axis_values = [sweep[k] for k in axis_keys]
    combos = list(itertools.product(*axis_values))
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f'[sweep] base config: {args.base_config}')
    print(f'[sweep] {len(combos)} runs planned on split={args.split} ({args.split_family})')
    for c in combos:
        axes = dict(zip(axis_keys, c))
        print(f'  - {axes}')

    start = datetime.now()
    summary: List[Dict[str, Any]] = []
    for combo in combos:
        axes = dict(zip(axis_keys, combo))
        run_name, run_dir, rc, metrics = _launch_run(base_cfg, axes, output_root, args)
        summary.append({
            'run_name': run_name,
            'axes': axes,
            'run_dir': str(run_dir),
            'returncode': rc,
            'metrics': metrics,
        })
        # Incremental save so we don't lose progress if a later run crashes.
        (output_root / 'sweep_summary.json').write_text(
            json.dumps({
                'base_config': str(args.base_config),
                'split': args.split,
                'split_family': args.split_family,
                'primary_metric': args.primary_metric,
                'runs': summary,
            }, indent=2)
        )

    # Rank + print table.
    primary = args.primary_metric
    ranked = sorted(
        [r for r in summary if primary in r.get('metrics', {})],
        key=lambda r: r['metrics'][primary],
    )
    print('\n' + '=' * 72)
    print(f'[sweep] finished {len(summary)} runs in {datetime.now() - start}')
    print(f'[sweep] ranking by val_losses["{primary}"] (lower = better)')
    print('=' * 72)
    for i, r in enumerate(ranked, 1):
        axes_str = ', '.join(f'{k}={v}' for k, v in r['axes'].items())
        m = r['metrics']
        metric_str = (
            f'{primary}={m.get(primary, float("nan")):.4f}  '
            f'total={m.get("total", float("nan")):.4f}  '
            f'kin={m.get("kin", float("nan")):.4f}  '
            f'gesture={m.get("gesture", float("nan")):.4f}  '
            f'skill={m.get("skill", float("nan")):.4f}'
        )
        print(f'  {i:>2}. {axes_str}')
        print(f'      {metric_str}')

    # Runs with no metrics (likely failures).
    failed = [r for r in summary if not r.get('metrics') or r.get('returncode', 0) != 0]
    if failed:
        print(f'\n[sweep] {len(failed)} runs did not produce metrics:')
        for r in failed:
            print(f'  - {r["run_name"]} (rc={r["returncode"]}, dir={r["run_dir"]})')

    print(f'\n[sweep] summary: {output_root / "sweep_summary.json"}')
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
