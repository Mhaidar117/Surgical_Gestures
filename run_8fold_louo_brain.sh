#!/bin/bash
#
# Run 8-fold LOUO cross-validation with brain alignment (eye-tracking task-centroid RSA).
# Trains on all 3 tasks per fold, evaluates per-task on each task's test set.
#
# Usage:
#   ./run_8fold_louo_brain.sh              # Run all folds
#   ./run_8fold_louo_brain.sh 1 3          # Run folds 1-3 only
#

set -e  # Exit on error

# Configuration
START_FOLD=${1:-1}
END_FOLD=${2:-8}
CONFIG="src/configs/brain_eye.yaml"
DATA_ROOT="."
BASE_OUTPUT_DIR="checkpoints/brain_eye"
EVAL_OUTPUT_DIR="eval_results/brain_eye"

echo "============================================================"
echo "8-Fold LOUO with Brain Alignment (Eye-Tracking RSA)"
echo "============================================================"
echo "Folds: $START_FOLD to $END_FOLD"
echo "Config: $CONFIG"
echo "Output: $BASE_OUTPUT_DIR"
echo "============================================================"

# Create output directories
mkdir -p "$EVAL_OUTPUT_DIR"

# Train and evaluate each fold (multi-task: all 3 tasks per fold)
for FOLD in $(seq $START_FOLD $END_FOLD); do
    SPLIT="fold_${FOLD}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${SPLIT}"

    echo ""
    echo "============================================================"
    echo "Fold $FOLD: Training on all 3 tasks"
    echo "============================================================"

    # Train on all tasks with brain alignment
    python3 src/training/train_vit_system.py \
        --config "$CONFIG" \
        --data_root "$DATA_ROOT" \
        --task all \
        --split "$SPLIT" \
        --output_dir "$OUTPUT_DIR" \
        --arm PSM2

    echo ""
    echo "Evaluating fold $FOLD on each task's test set..."

    # Evaluate on each task's test set (model was trained on all 3)
    for TASK in Knot_Tying Needle_Passing Suturing; do
        echo "  - $TASK..."
        python3 src/eval/evaluate.py \
            --checkpoint "$OUTPUT_DIR/best_model.pth" \
            --data_root "$DATA_ROOT" \
            --task "$TASK" \
            --split "$SPLIT" \
            --mode test 2>&1 | tee "$EVAL_OUTPUT_DIR/${TASK}_test_${SPLIT}.txt"
    done

    echo "- Completed fold $FOLD"
done

echo ""
echo "============================================================"
echo "All brain-aligned training and evaluation complete!"
echo "============================================================"
echo ""
echo "Results saved to: $EVAL_OUTPUT_DIR/"
echo ""
echo "To aggregate results, run:"
echo "  python3 aggregate_louo_results.py --eval_dir $EVAL_OUTPUT_DIR"
