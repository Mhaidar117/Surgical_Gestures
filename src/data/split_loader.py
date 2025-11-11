"""
LOUO (Leave-One-User-Out) split management for JIGSAWS dataset.
"""
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def generate_louo_splits(
    data_root: str,
    task: str = 'Knot_Tying',
    output_dir: Optional[str] = None
) -> Dict[str, Dict[str, List[str]]]:
    """
    Generate 8-fold LOUO splits based on surgeon IDs.
    
    Args:
        data_root: Root directory containing Gestures/ folder
        task: Task name
        output_dir: Directory to save split files, None uses data_root/data/splits/
    
    Returns:
        Dictionary mapping fold names to train/val/test trial lists
    """
    data_root = Path(data_root)
    meta_file = data_root / 'Gestures' / task / f'meta_file_{task}.txt'
    
    # Group trials by surgeon ID
    surgeon_trials = defaultdict(list)
    
    if meta_file.exists():
        with open(meta_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    trial_id = parts[0]
                    surgeon_id = parts[1] if len(parts) > 1 else 'Unknown'
                    surgeon_trials[surgeon_id].append(trial_id)
    
    # Get unique surgeon IDs
    surgeon_ids = sorted(surgeon_trials.keys())
    
    # Generate 8 folds (leave one surgeon out per fold)
    splits = {}
    
    for i, test_surgeon in enumerate(surgeon_ids):
        fold_name = f'fold_{i+1}'
        
        # Test set: all trials from test_surgeon
        test_trials = surgeon_trials[test_surgeon]
        
        # Validation set: trials from one other surgeon (if available)
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
            'val_surgeon': val_surgeon
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
    
    # Save as YAML
    yaml_file = output_dir / f'{task}_splits.yaml'
    with open(yaml_file, 'w') as f:
        yaml.dump(splits, f, default_flow_style=False)
    
    print(f"Generated {len(splits)} LOUO splits for {task}")
    print(f"Saved to: {json_file} and {yaml_file}")
    
    return splits


def load_splits(
    split_file: str,
    fold_name: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Load split definitions from file.
    
    Args:
        split_file: Path to split file (JSON or YAML)
        fold_name: Name of fold to load, None returns all folds
    
    Returns:
        Dictionary with train/val/test trial lists
    """
    split_file = Path(split_file)
    
    if split_file.suffix == '.json':
        with open(split_file, 'r') as f:
            splits = json.load(f)
    elif split_file.suffix in ['.yaml', '.yml']:
        with open(split_file, 'r') as f:
            splits = yaml.safe_load(f)
    else:
        raise ValueError(f"Unknown file format: {split_file.suffix}")
    
    if fold_name is not None:
        if fold_name in splits:
            return splits[fold_name]
        else:
            raise ValueError(f"Fold {fold_name} not found in splits")
    
    return splits


def get_trials_for_split(
    data_root: str,
    task: str,
    split_name: str,
    mode: str = 'train'
) -> List[str]:
    """
    Get trial IDs for a specific split and mode.
    
    Args:
        data_root: Root directory
        task: Task name
        split_name: Split name (e.g., 'fold_1')
        mode: 'train', 'val', or 'test'
    
    Returns:
        List of trial IDs
    """
    data_root = Path(data_root)
    split_file = data_root / 'data' / 'splits' / f'{task}_splits.json'
    
    if not split_file.exists():
        # Generate splits if they don't exist
        splits = generate_louo_splits(data_root, task)
    else:
        splits = load_splits(str(split_file))
    
    if split_name not in splits:
        raise ValueError(f"Split {split_name} not found")
    
    return splits[split_name].get(mode, [])


class SplitLoader:
    """
    Convenience class for loading and managing data splits.
    """
    
    def __init__(
        self,
        data_root: str,
        task: str = 'Knot_Tying',
        split_name: Optional[str] = None
    ):
        """
        Args:
            data_root: Root directory
            task: Task name
            split_name: Split name (e.g., 'fold_1'), None uses all data
        """
        self.data_root = Path(data_root)
        self.task = task
        self.split_name = split_name
        
        # Load or generate splits
        split_file = self.data_root / 'data' / 'splits' / f'{task}_splits.json'
        if split_file.exists():
            self.splits = load_splits(str(split_file))
        else:
            self.splits = generate_louo_splits(str(data_root), task)
        
        if split_name is not None:
            if split_name not in self.splits:
                raise ValueError(f"Split {split_name} not found")
            self.current_split = self.splits[split_name]
        else:
            self.current_split = None
    
    def get_train_trials(self) -> List[str]:
        """Get training trial IDs."""
        if self.current_split:
            return self.current_split['train']
        else:
            # Return all trials from all splits
            all_trials = set()
            for split in self.splits.values():
                all_trials.update(split['train'])
            return sorted(list(all_trials))
    
    def get_val_trials(self) -> List[str]:
        """Get validation trial IDs."""
        if self.current_split:
            return self.current_split['val']
        else:
            all_trials = set()
            for split in self.splits.values():
                all_trials.update(split['val'])
            return sorted(list(all_trials))
    
    def get_test_trials(self) -> List[str]:
        """Get test trial IDs."""
        if self.current_split:
            return self.current_split['test']
        else:
            all_trials = set()
            for split in self.splits.values():
                all_trials.update(split['test'])
            return sorted(list(all_trials))
    
    def get_all_folds(self) -> List[str]:
        """Get list of all fold names."""
        return sorted(self.splits.keys())

