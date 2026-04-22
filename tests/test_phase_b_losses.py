"""Tests for Phase B loss primitives.

Covers:
  1. ``pairwise_distance_rdm`` — shape, symmetry, zero diagonal, differentiable.
  2. ``kinematics_trajectory_features`` — shape, masked pooling.
  3. ``surgeon_conditioned_skill_contrastive_loss`` — surgeon masking respected,
     cross-surgeon-same-skill pair pulls positive gradient.
  4. ``surgeon_ids_from_trial_ids`` — JIGSAWS ID parsing.
  5. ``compute_total_loss`` with brain_mode='kinematics_rsa' produces a finite
     loss and with skill_contra weight > 0 includes the extra term.

Run:
    python -m pytest tests/test_phase_b_losses.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from modules.brain_rdm import (  # noqa: E402
    pairwise_distance_rdm,
    kinematics_trajectory_features,
)
from models.losses import (  # noqa: E402
    compute_total_loss,
    surgeon_conditioned_skill_contrastive_loss,
    surgeon_ids_from_trial_ids,
)


def test_pairwise_distance_rdm_shape_and_symmetry():
    feat = torch.randn(6, 8, requires_grad=True)
    rdm = pairwise_distance_rdm(feat)
    assert rdm.shape == (6, 6)
    assert torch.allclose(rdm, rdm.t(), atol=1e-5)
    assert torch.allclose(rdm.diagonal(), torch.zeros(6), atol=1e-3)

    # Differentiable
    rdm.sum().backward()
    assert feat.grad is not None
    assert feat.grad.abs().sum() > 0


def test_kinematics_trajectory_features_shape():
    kin = torch.randn(4, 10, 19)
    feat = kinematics_trajectory_features(kin)
    assert feat.shape == (4, 38)  # 2 * K
    # mean_i is in feat[:, :K]; std_i is in feat[:, K:]
    assert torch.allclose(feat[:, :19], kin.mean(dim=1))


def test_kinematics_trajectory_features_masked():
    kin = torch.zeros(2, 6, 3)
    kin[:, :3, :] = 1.0  # first half = 1; second half = 0
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[:, 3:] = False  # ignore second half

    feat = kinematics_trajectory_features(kin, mask=mask)
    # masked mean over 1.0s == 1.0
    assert torch.allclose(feat[:, :3], torch.ones(2, 3), atol=1e-5)


def test_surgeon_ids_parser():
    tids = ['Knot_Tying_B001', 'Knot_Tying_C002', 'Knot_Tying_B003', 'weird']
    out = surgeon_ids_from_trial_ids(tids)
    assert out.shape == (4,)
    # B gets first id (0), C gets second (1), weird -> -1
    assert out[0].item() == out[2].item()  # same surgeon
    assert out[0].item() != out[1].item()
    assert out[3].item() == -1


def test_skill_contrastive_masks_same_surgeon():
    # 4 samples: all same surgeon. No cross-surgeon pairs → loss should be 0.
    embs = torch.randn(4, 16, requires_grad=True)
    skills = torch.tensor([0, 0, 1, 1])
    surgeons = torch.tensor([0, 0, 0, 0])
    loss = surgeon_conditioned_skill_contrastive_loss(embs, skills, surgeons)
    assert loss.item() == 0.0


def test_skill_contrastive_active_when_cross_surgeon_pairs_exist():
    # 4 samples, 2 surgeons, each skill represented in both surgeons -> pairs exist.
    torch.manual_seed(0)
    embs = torch.randn(4, 16, requires_grad=True)
    skills = torch.tensor([0, 1, 0, 1])
    surgeons = torch.tensor([0, 0, 1, 1])
    loss = surgeon_conditioned_skill_contrastive_loss(embs, skills, surgeons)
    assert loss.item() > 0
    loss.backward()
    assert embs.grad is not None and embs.grad.abs().sum() > 0


def test_skill_contrastive_reduces_with_alignment():
    """Loss should decrease monotonically as the same-skill-cross-surgeon
    positive moves closer to the anchor (starting from misaligned)."""
    torch.manual_seed(42)
    anchor = torch.tensor([[1.0, 0.0]])        # skill 0, surgeon 0
    neg = torch.tensor([[-1.0, 0.0]])          # skill 1, surgeon 1 (should be pushed away)
    skills = torch.tensor([0, 0, 1])
    surgeons = torch.tensor([0, 1, 1])

    losses = []
    for alpha in [0.0, 0.4, 0.8]:
        # Vary positive: alpha=0 is orthogonal to anchor, alpha=1 is aligned.
        pos = torch.tensor([[alpha, 1.0 - alpha]])
        embs = torch.cat([anchor, pos, neg])
        loss = surgeon_conditioned_skill_contrastive_loss(
            embs, skills, surgeons, temperature=0.1
        )
        losses.append(loss.item())
    # Loss should drop as positive aligns with anchor.
    assert losses[0] > losses[1] > losses[2], f'Expected monotone decrease, got {losses}'


def test_compute_total_loss_kinematics_rsa_mode():
    B, T, K = 4, 5, 19
    pred_kin = torch.randn(B, T, K, requires_grad=True)
    tgt_kin = torch.randn(B, T, K)
    gesture_logits = torch.randn(B, 15, requires_grad=True)
    gesture_labels = torch.randint(0, 15, (B,))
    skill_logits = torch.randn(B, 3, requires_grad=True)
    skill_labels = torch.randint(0, 3, (B,))

    # Build a fake pair of RDMs via the real primitive.
    model_emb = torch.randn(B, 32, requires_grad=True)
    kin_feat = kinematics_trajectory_features(tgt_kin)
    model_rdm = pairwise_distance_rdm(model_emb)
    target_rdm = pairwise_distance_rdm(kin_feat).detach()

    total, comps = compute_total_loss(
        pred_kin, tgt_kin,
        gesture_logits, gesture_labels,
        skill_logits, skill_labels,
        model_rdm=model_rdm,
        eeg_rdm=target_rdm,
        brain_mode='kinematics_rsa',
        loss_weights={'kin': 1.0, 'gesture': 1.0, 'skill': 0.5,
                      'brain': 0.1, 'control': 0.0},
    )
    assert torch.isfinite(total)
    assert 'brain_rsa' in comps
    total.backward()
    assert model_emb.grad is not None
    assert model_emb.grad.abs().sum() > 0


def test_compute_total_loss_skill_contra_adds_term():
    B = 6
    pred_kin = torch.randn(B, 4, 19, requires_grad=True)
    tgt_kin = torch.randn(B, 4, 19)
    gesture_logits = torch.randn(B, 15, requires_grad=True)
    gesture_labels = torch.randint(0, 15, (B,))
    skill_logits = torch.randn(B, 3, requires_grad=True)
    skill_labels = torch.tensor([0, 0, 1, 1, 2, 2])
    surgeon_ids = torch.tensor([0, 1, 0, 1, 0, 1])
    skill_embs = torch.randn(B, 32, requires_grad=True)

    # With weight > 0 the term appears in comps.
    total, comps = compute_total_loss(
        pred_kin, tgt_kin,
        gesture_logits, gesture_labels,
        skill_logits, skill_labels,
        brain_mode='none',
        loss_weights={'kin': 1.0, 'gesture': 1.0, 'skill': 0.5,
                      'brain': 0.0, 'control': 0.0,
                      'skill_contra': 0.1},
        skill_embeddings=skill_embs,
        surgeon_ids=surgeon_ids,
    )
    assert 'skill_contra' in comps
    assert torch.isfinite(total)
    total.backward()
    assert skill_embs.grad is not None
    assert skill_embs.grad.abs().sum() > 0


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
