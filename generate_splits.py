#!/usr/bin/env python
"""
Generate LOUO (Leave-One-User-Out) splits for JIGSAWS dataset.

Usage:
    python generate_splits.py
"""
import os
import json
from pathlib import Path
from collections import defaultdict

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def generate_louo_splits(data_root, task, output_dir=None):
    """Generate 8-fold LOUO splits based on surgeon IDs."""
    data_root = Path(data_root)
    meta_file = data_root / 'Gestures' / task / f'meta_file_{task}.txt'

    # Group trials by surgeon ID
    surgeon_trials = defaultdict(list)
    skill_levels = {}  # Track skill level per surgeon

    if meta_file.exists():
        with open(meta_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 1:
                    trial_id = parts[0].strip()
                    # Skip empty trial IDs
                    if not trial_id:
                        continue
                    # Extract surgeon ID from trial name (e.g., "Knot_Tying_B001" -> "B")
                    trial_suffix = trial_id.replace(f'{task}_', '')
                    if not trial_suffix:
                        continue
                    surgeon_id = trial_suffix[0]
                    # Skip invalid surgeon IDs
                    if not surgeon_id.isalpha():
                        continue
                    surgeon_trials[surgeon_id].append(trial_id)

                    # Extract skill level (N=Novice, I=Intermediate, E=Expert)
                    if len(parts) >= 3 and parts[2] in ['N', 'I', 'E']:
                        skill_levels[surgeon_id] = parts[2]
    else:
        print(f"Warning: meta file not found: {meta_file}")
        return {}

    # Get unique surgeon IDs (only valid alphabetic IDs)
    surgeon_ids = sorted([s for s in surgeon_trials.keys() if s.isalpha()])
    print(f"Found {len(surgeon_ids)} surgeons: {surgeon_ids}")
    print(f"Trials per surgeon: {dict((s, len(surgeon_trials[s])) for s in surgeon_ids)}")
    if skill_levels:
        print(f"Skill levels: {skill_levels}")

    # Generate folds (leave one surgeon out per fold)
    splits = {}

    for i, test_surgeon in enumerate(surgeon_ids):
        fold_name = f'fold_{i+1}'

        # Test set: all trials from test_surgeon
        test_trials = surgeon_trials[test_surgeon]

        # Validation set: trials from one other surgeon
        val_surgeon = surgeon_ids[(i + 1) % len(surgeon_ids)]
        val_trials = surgeon_trials[val_surgeon]

        # Training set: all other surgeons
        train_trials = []
        for surgeon_id in surgeon_ids:
            if surgeon_id != test_surgeon and surgeon_id != val_surgeon:
                train_trials.extend(surgeon_trials[surgeon_id])

        splits[fold_name] = {
            'train': sorted(train_trials),
            'val': sorted(val_trials),
            'test': sorted(test_trials),
            'test_surgeon': test_surgeon,
            'val_surgeon': val_surgeon,
            'test_skill': skill_levels.get(test_surgeon, 'Unknown'),
            'val_skill': skill_levels.get(val_surgeon, 'Unknown')
        }

    # Save splits
    if output_dir is None:
        output_dir = data_root / 'data' / 'splits'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    json_file = output_dir / f'{task}_splits.json'
    with open(json_file, 'w') as f:
        json.dump(splits, f, indent=2)

    # Save as YAML (optional)
    if HAS_YAML:
        yaml_file = output_dir / f'{task}_splits.yaml'
        with open(yaml_file, 'w') as f:
            yaml.dump(splits, f, default_flow_style=False)

    print(f"Generated {len(splits)} LOUO splits for {task}")
    print(f"Saved to: {json_file}")

    return splits


def print_split_summary(splits, task):
    """Print a summary table of all splits."""
    print(f"\n{'Fold':<8} {'Train':<8} {'Val':<8} {'Test':<8} {'Test Surgeon':<15} {'Val Surgeon':<15}")
    print("-" * 70)

    total_train, total_val, total_test = 0, 0, 0
    for fold, data in sorted(splits.items()):
        test_info = f"{data['test_surgeon']} ({data.get('test_skill', '?')})"
        val_info = f"{data['val_surgeon']} ({data.get('val_skill', '?')})"
        print(f"{fold:<8} {len(data['train']):<8} {len(data['val']):<8} {len(data['test']):<8} {test_info:<15} {val_info:<15}")
        total_train += len(data['train'])
        total_val += len(data['val'])
        total_test += len(data['test'])

    print("-" * 70)
    n_folds = len(splits)
    print(f"{'Avg':<8} {total_train/n_folds:<8.1f} {total_val/n_folds:<8.1f} {total_test/n_folds:<8.1f}")


if __name__ == '__main__':
    # Generate splits for all tasks
    all_splits = {}

    for task in ['Knot_Tying', 'Needle_Passing', 'Suturing']:
        print(f'\n{"="*70}')
        print(f' {task}')
        print(f'{"="*70}')

        try:
            splits = generate_louo_splits('.', task)
            all_splits[task] = splits
            print_split_summary(splits, task)
        except Exception as e:
            import traceback
            print(f'Error: {e}')
            traceback.print_exc()

    print(f'\n{"="*70}')
    print(" Summary: Splits saved to data/splits/")
    print(f'{"="*70}')
    for task in all_splits:
        print(f"  - {task}_splits.json")
        print(f"  - {task}_splits.yaml")
