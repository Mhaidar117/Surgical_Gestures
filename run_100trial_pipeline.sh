#!/usr/bin/env bash
# ============================================================
# run_100trial_pipeline.sh
# Runs the EEG–eye bridge pipeline (Phases 1–3) on 100 real
# trials and writes results to cache/eeg_eye_bridge/.
#
# Usage:
#   cd /path/to/Surgical_Gestures
#   bash run_100trial_pipeline.sh
# ============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "==> Repo root: $REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src"

# ---------- Phase 1: EEG encoding (100 trials) ----------
echo ""
echo "============================================================"
echo "  PHASE 1: Load EDF → filter → window → encode (100 trials)"
echo "============================================================"
python3 scripts/eeg_eye_bridge/phase1/run_export.py \
    --data_root "$REPO_ROOT" \
    --max_trials 100 \
    --window_sec 1.0 \
    --hop_sec 0.5 \
    --device cpu

echo ""
echo "==> Phase 1 complete."

# ---------- Phase 2: Eye-consistency (100-trial subset) ----------
echo ""
echo "============================================================"
echo "  PHASE 2: Eye summaries + EEG–eye consistency scoring"
echo "============================================================"
python3 scripts/eeg_eye_bridge/phase2/run_phase2.py \
    --repo-root "$REPO_ROOT" \
    --subset 100

echo ""
echo "==> Phase 2 complete."

# ---------- Phase 3: RDM construction ----------
echo ""
echo "============================================================"
echo "  PHASE 3: Build candidate RDMs + write manifest"
echo "============================================================"
python3 scripts/eeg_eye_bridge/phase3/build_rdms.py \
    --cache-root "$REPO_ROOT/cache/eeg_eye_bridge"

echo ""
echo "==> Phase 3 complete."

echo ""
echo "============================================================"
echo "  ALL PHASES DONE"
echo "  Results in: cache/eeg_eye_bridge/"
echo "    Phase 1 trials : cache/eeg_eye_bridge/phase1/trials/   (100 PKLs)"
echo "    Phase 2 scores : cache/eeg_eye_bridge/phase2/eye_consistency_scores.pkl"
echo "    Phase 3 RDMs   : cache/eeg_eye_bridge/phase3/rdms/"
echo "    RDM manifest   : cache/eeg_eye_bridge/phase3/rdm_manifest.json"
echo "============================================================"
