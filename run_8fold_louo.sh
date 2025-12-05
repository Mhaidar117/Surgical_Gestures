#!/bin/bash
#
# Run 8-fold LOUO cross-validation for JIGSAWS tasks
#
# This script trains and evaluates models for all folds of a given task,
# then aggregates results to compute mean ± std metrics.
#
# Usage:
#   ./run_8fold_louo.sh                    # Run all tasks with all folds
#   ./run_8fold_louo.sh Knot_Tying         # Run only Knot_Tying
#   ./run_8fold_louo.sh Knot_Tying 1 3     # Run Knot_Tying folds 1-3 only
#

set -e  # Exit on error

# Configuration
TASK_FILTER=${1:-all}
START_FOLD=${2:-1}
END_FOLD=${3:-8}
CONFIG="src/configs/baseline.yaml"
DATA_ROOT="."
BASE_OUTPUT_DIR="checkpoints"
EVAL_OUTPUT_DIR="eval_results"

echo "============================================================"
echo "8-Fold LOUO Cross-Validation"
echo "============================================================"
echo "Task filter: $TASK_FILTER"
echo "Folds: $START_FOLD to $END_FOLD"
echo "Config: $CONFIG"
echo "============================================================"

# Create output directories
mkdir -p "$EVAL_OUTPUT_DIR"

# Define tasks
if [ "$TASK_FILTER" = "all" ]; then
    TASKS=("Knot_Tying" "Needle_Passing" "Suturing")
else
    TASKS=("$TASK_FILTER")
fi

# Get number of folds per task from splits
get_num_folds() {
    local task=$1
    local splits_file="data/splits/${task}_splits.json"
    if [ -f "$splits_file" ]; then
        # Count fold entries in JSON
        grep -c '"fold_' "$splits_file" || echo "8"
    else
        echo "8"
    fi
}

# Train and evaluate each task
for TASK in "${TASKS[@]}"; do
    TASK_LOWER=$(echo "$TASK" | tr '[:upper:]' '[:lower:]')
    NUM_FOLDS=$(get_num_folds "$TASK")

    # Adjust end fold if task has fewer folds
    ACTUAL_END_FOLD=$END_FOLD
    if [ "$END_FOLD" -gt "$NUM_FOLDS" ]; then
        ACTUAL_END_FOLD=$NUM_FOLDS
    fi

    echo ""
    echo "============================================================"
    echo "Task: $TASK ($NUM_FOLDS folds available)"
    echo "Running folds: $START_FOLD to $ACTUAL_END_FOLD"
    echo "============================================================"

    # Train and evaluate each fold
    for FOLD in $(seq $START_FOLD $ACTUAL_END_FOLD); do
        SPLIT="fold_${FOLD}"
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TASK_LOWER}_${SPLIT}"

        echo ""
        echo "------------------------------------------------------------"
        echo "Training: $TASK - $SPLIT"
        echo "Output: $OUTPUT_DIR"
        echo "------------------------------------------------------------"

        # Train
        python3 src/training/train_vit_system.py \
            --config "$CONFIG" \
            --data_root "$DATA_ROOT" \
            --task "$TASK" \
            --split "$SPLIT" \
            --output_dir "$OUTPUT_DIR" \
            --arm PSM2

        echo ""
        echo "Evaluating: $TASK - $SPLIT"

        # Evaluate on test set
        python3 src/eval/evaluate.py \
            --checkpoint "$OUTPUT_DIR/best_model.pth" \
            --data_root "$DATA_ROOT" \
            --task "$TASK" \
            --split "$SPLIT" \
            --mode test \
            --output_dir "$EVAL_OUTPUT_DIR"

        echo "✓ Completed $TASK - $SPLIT"
    done

    echo ""
    echo "============================================================"
    echo "Completed all folds for $TASK"
    echo "============================================================"
done

echo ""
echo "============================================================"
echo "All training and evaluation complete!"
echo "============================================================"
echo ""
echo "Results saved to: $EVAL_OUTPUT_DIR/"
echo ""
echo "To aggregate results across folds, run:"
echo "  python3 aggregate_louo_results.py"
