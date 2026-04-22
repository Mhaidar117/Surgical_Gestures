#!/bin/bash
#
# Production bridge-RDM training + evaluation for fold 1.
# Uses: joint EEG+Eye RDM (joint_eye_eeg_task_family) as the RSA regularizer.
#
# Usage (from the Surgical_Gestures directory):
#   chmod +x run_bridge_production_fold1.sh
#   ./run_bridge_production_fold1.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH=src
# upsample_linear1d is not yet implemented on MPS; fall back to CPU for that op.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Use the venv's Python explicitly so av, torch, etc. are always found
# regardless of which python3 is on the system PATH.
PYTHON="${SCRIPT_DIR}/.venv/bin/python"
if [ ! -f "${PYTHON}" ]; then
    echo "ERROR: .venv not found at ${SCRIPT_DIR}/.venv — run: python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

CONFIG="src/configs/bridge_production.yaml"
SPLIT="fold_1"
OUTPUT_DIR="checkpoints/bridge_production/fold_1"
EVAL_DIR="eval_results/bridge_production"

mkdir -p "$OUTPUT_DIR" "$EVAL_DIR"

echo "============================================================"
echo "Bridge-RDM Production Run — Fold 1"
echo "Config  : $CONFIG  (joint EEG+Eye RDM, brain_mode=bridge)"
echo "Split   : $SPLIT"
echo "Tasks   : all (Knot_Tying, Needle_Passing, Suturing)"
echo "Output  : $OUTPUT_DIR"
echo "============================================================"
echo ""

# ── Phase 4: Train ────────────────────────────────────────────────
echo "[ Training ]"
"${PYTHON}" src/training/train_vit_system.py \
    --config "$CONFIG" \
    --data_root . \
    --task all \
    --split "$SPLIT" \
    --output_dir "$OUTPUT_DIR" \
    --arm PSM2

echo ""
echo "[ Evaluation ]"

# ── Evaluate per task ─────────────────────────────────────────────
for TASK in Knot_Tying Needle_Passing Suturing; do
    echo "  Evaluating $TASK ..."
    "${PYTHON}" src/eval/evaluate.py \
        --checkpoint "$OUTPUT_DIR/best_model.pth" \
        --data_root . \
        --task "$TASK" \
        --split "$SPLIT" \
        --mode test \
        2>&1 | tee "$EVAL_DIR/${TASK}_test_${SPLIT}.txt"
done

echo ""
echo "============================================================"
echo "Done! Results saved to: $EVAL_DIR/"
echo ""
echo "Compare against baseline (eval_results/*.txt):"
echo "  diff eval_results/Suturing_test_fold_1_results.txt \\"
echo "       $EVAL_DIR/Suturing_test_fold_1.txt"
echo "============================================================"
