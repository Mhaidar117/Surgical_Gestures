"""Unit tests for the three split families.

Constructs a tiny fake JIGSAWS layout under a tmp dir (meta_file + transcription
files only) and verifies the generator output for each family plus the
SplitLoader / filter plumbing.

Run:
    python -m pytest tests/test_split_families.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'pipeline'))

from generate_splits import (  # type: ignore
    generate_louo_splits,
    generate_inter_trial_within_subject_splits,
    generate_intra_trial_half_splits,
)
from data.split_loader import SplitLoader  # type: ignore


TASK = 'Knot_Tying'
SURGEONS = ['B', 'C', 'D']
TRIALS_PER_SURGEON = 4


def _make_fake_repo(tmp_path: Path) -> Path:
    """Populate a fake JIGSAWS layout with 3 surgeons x 4 trials.

    Each trial has 3 gesture segments spanning frames [0, 600].
    """
    task_dir = tmp_path / 'Gestures' / TASK
    (task_dir / 'transcriptions').mkdir(parents=True)

    meta_lines = []
    for s in SURGEONS:
        for n in range(1, TRIALS_PER_SURGEON + 1):
            trial_id = f'{TASK}_{s}{n:03d}'
            meta_lines.append(f'{trial_id}\t{s}\tE')
            trans = task_dir / 'transcriptions' / f'{trial_id}.txt'
            trans.write_text('0 200 G1\n200 400 G2\n400 600 G3\n')

    (task_dir / f'meta_file_{TASK}.txt').write_text('\n'.join(meta_lines) + '\n')
    return tmp_path


def test_louo_baseline(tmp_path):
    repo = _make_fake_repo(tmp_path)
    splits = generate_louo_splits(str(repo), TASK)

    assert len(splits) == len(SURGEONS)
    for fold_name, fold in splits.items():
        # Test surgeon's trials should not appear in train.
        test_surgeon = fold['test_surgeon']
        for t in fold['train']:
            suffix = t.replace(f'{TASK}_', '')
            assert suffix[0] != test_surgeon


def test_inter_trial_within_subject(tmp_path):
    repo = _make_fake_repo(tmp_path)
    splits = generate_inter_trial_within_subject_splits(str(repo), TASK)

    fold_keys = [k for k in splits if k.startswith('fold_')]
    assert len(fold_keys) == len(SURGEONS)
    assert splits['split_family'] == 'inter_trial_within_subject'

    for fold_name in fold_keys:
        fold = splits[fold_name]
        # Exactly one held-out trial.
        assert len(fold['test']) == 1
        held = fold['test'][0]
        test_surgeon = fold['test_surgeon']

        # Held-out trial belongs to the named test_surgeon.
        suffix = held.replace(f'{TASK}_', '')
        assert suffix[0] == test_surgeon

        # Train contains other trials from that same surgeon (key contrast with LOUO).
        same_surgeon_train = [
            t for t in fold['train'] if t.replace(f'{TASK}_', '')[0] == test_surgeon
        ]
        assert len(same_surgeon_train) == TRIALS_PER_SURGEON - 1

        # Held-out trial never in train.
        assert held not in fold['train']


def test_intra_trial_half(tmp_path):
    repo = _make_fake_repo(tmp_path)
    splits = generate_intra_trial_half_splits(str(repo), TASK)

    fold_keys = [k for k in splits if k.startswith('fold_')]
    assert len(fold_keys) == len(SURGEONS)
    assert splits['split_family'] == 'intra_trial_half'

    for fold_name in fold_keys:
        fold = splits[fold_name]
        assert fold['train'] == fold['test']  # same trials; split by frame
        sf = fold['segment_filter']
        assert 'train' in sf and 'test' in sf

        for trial_id, bounds in sf['train'].items():
            assert 'end_frame_max' in bounds
            # Our fake transcription spans 0..600 -> midpoint 300.
            assert bounds['end_frame_max'] == 300

        for trial_id, bounds in sf['test'].items():
            assert 'start_frame_min' in bounds
            assert bounds['start_frame_min'] == 300


def test_split_loader_segment_filter(tmp_path):
    repo = _make_fake_repo(tmp_path)
    generate_intra_trial_half_splits(str(repo), TASK)

    sl = SplitLoader(str(repo), task=TASK, split_name='fold_1',
                     split_family='intra_trial_half')
    train_sf = sl.get_segment_filter('train')
    test_sf = sl.get_segment_filter('test')

    assert train_sf is not None
    assert test_sf is not None
    for bounds in train_sf.values():
        assert bounds['end_frame_max'] == 300
    for bounds in test_sf.values():
        assert bounds['start_frame_min'] == 300


def test_split_loader_rejects_unknown_family(tmp_path):
    repo = _make_fake_repo(tmp_path)
    generate_louo_splits(str(repo), TASK)
    with pytest.raises(ValueError, match='Unknown split_family'):
        SplitLoader(str(repo), task=TASK, split_name='fold_1',
                    split_family='bogus')


def test_filter_dataset_by_trials_respects_bounds(tmp_path):
    """filter_dataset_by_trials honors end_frame_max / start_frame_min."""
    from training.train_vit_system import filter_dataset_by_trials  # type: ignore

    class _FakeDataset:
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    trial_id = f'{TASK}_B001'
    samples = [
        {'trial_id': trial_id, 'start_frame': 0, 'end_frame': 100},
        {'trial_id': trial_id, 'start_frame': 100, 'end_frame': 200},
        {'trial_id': trial_id, 'start_frame': 200, 'end_frame': 300},
        {'trial_id': trial_id, 'start_frame': 300, 'end_frame': 400},
        {'trial_id': trial_id, 'start_frame': 400, 'end_frame': 500},
    ]
    ds = _FakeDataset(samples)

    train_filter = {trial_id: {'end_frame_max': 300}}
    train_sub = filter_dataset_by_trials(ds, [trial_id], segment_filter=train_filter)
    # Segments with end_frame <= 300: first three.
    assert len(train_sub) == 3

    test_filter = {trial_id: {'start_frame_min': 300}}
    test_sub = filter_dataset_by_trials(ds, [trial_id], segment_filter=test_filter)
    # Segments with start_frame >= 300: last two.
    assert len(test_sub) == 2


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
