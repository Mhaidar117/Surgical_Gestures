#!/usr/bin/env bash
# ============================================================
# run_trained_pipeline.sh
# Full pipeline with self-supervised model training:
#
#   Step 1 — Train Baseline CNN + PC GRU on 100 EDF trials
#   Step 2 — Re-export Phase 1 using trained weights
#   Step 3 — Phase 2: eye summaries + eye-consistency scoring
#   Step 4 — Phase 3: build candidate RDMs + write manifest
#
# Usage:
#   cd /path/to/Surgical_Gestures
#   source ~/eeg_venv/bin/activate
#   bash run_trained_pipeline.sh
#
# Flags:
#   --epochs N        training epochs per model (default: 20)
#   --max_trials N    number of trials to use    (default: 100)
# ============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "==> Repo root: $REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src"

# Parse optional flags
EPOCHS=20
MAX_TRIALS=100
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)    EPOCHS="$2";     shift 2 ;;
        --max_trials) MAX_TRIALS="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

BASELINE_CKPT="$REPO_ROOT/checkpoints/eeg_models/baseline.pt"
PC_CKPT="$REPO_ROOT/checkpoints/eeg_models/pc_model.pt"

# ---------- Step 1: Train ----------
echo ""
echo "============================================================"
echo "  STEP 1: Self-supervised training ($EPOCHS epochs, $MAX_TRIALS trials)"
echo "============================================================"
python3 scripts/eeg_eye_bridge/train_eeg_models.py \
    --data_root    "$REPO_ROOT" \
    --max_trials   "$MAX_TRIALS" \
    --epochs       "$EPOCHS"

echo ""
echo "==> Training complete. Weights saved to checkpoints/eeg_models/"

# ---------- Step 2: Phase 1 with trained weights ----------
echo ""
echo "============================================================"
echo "  STEP 2: Phase 1 export with trained weights"
echo "============================================================"
python3 scripts/eeg_eye_bridge/phase1/run_export.py \
    --data_root      "$REPO_ROOT" \
    --max_trials     "$MAX_TRIALS" \
    --window_sec     1.0 \
    --hop_sec        0.5 \
    --device         cpu \
    --baseline_ckpt  "$BASELINE_CKPT" \
    --pc_ckpt        "$PC_CKPT"

echo ""
echo "==> Phase 1 complete."

# ---------- Step 3: Phase 2 ----------
echo ""
echo "============================================================"
echo "  STEP 3: Eye summaries + EEG–eye consistency scoring"
echo "============================================================"
python3 scripts/eeg_eye_bridge/phase2/run_phase2.py \
    --repo-root "$REPO_ROOT" \
    --subset    "$MAX_TRIALS"

echo ""
echo "==> Phase 2 complete."

# ---------- Step 4: Phase 3 ----------
echo ""
echo "============================================================"
echo "  STEP 4: Build candidate RDMs + write manifest"
echo "============================================================"
python3 scripts/eeg_eye_bridge/phase3/build_rdms.py \
    --cache-root "$REPO_ROOT/cache/eeg_eye_bridge"

echo ""
echo "==> Phase 3 complete."

echo ""
echo "============================================================"
echo "  ALL STEPS DONE"
echo "  Trained weights : checkpoints/eeg_models/"
echo "  Phase 1 trials  : cache/eeg_eye_bridge/phase1/trials/"
echo "  Phase 2 scores  : cache/eeg_eye_bridge/phase2/eye_consistency_scores.pkl"
echo "  Phase 3 RDMs    : cache/eeg_eye_bridge/phase3/rdms/"
echo "  RDM manifest    : cache/eeg_eye_bridge/phase3/rdm_manifest.json"
echo "============================================================"
