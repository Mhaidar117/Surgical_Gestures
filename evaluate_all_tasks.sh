#!/bin/bash
#
# Evaluate models for all three JIGSAWS tasks
#
# Usage:
#   ./evaluate_all_tasks.sh              # Evaluate all tasks with fold_1 on test set
#   ./evaluate_all_tasks.sh fold_2       # Evaluate all tasks with fold_2
#   ./evaluate_all_tasks.sh fold_1 val   # Evaluate on validation set
#

set -e  # Exit on error

# Configuration
SPLIT=${1:-fold_1}
MODE=${2:-test}
BASE_OUTPUT_DIR="checkpoints"
EVAL_OUTPUT_DIR="eval_results"

echo "============================================================"
echo "JIGSAWS Evaluation Pipeline"
echo "============================================================"
echo "Split: $SPLIT"
echo "Mode: $MODE"
echo "Results will be saved to: $EVAL_OUTPUT_DIR"
echo "============================================================"

# Create eval output directory
mkdir -p "$EVAL_OUTPUT_DIR"

# Define tasks
TASKS=("Knot_Tying" "Needle_Passing" "Suturing")

# Evaluate each task
for TASK in "${TASKS[@]}"; do
    TASK_LOWER=$(echo "$TASK" | tr '[:upper:]' '[:lower:]')
    CHECKPOINT="${BASE_OUTPUT_DIR}/${TASK_LOWER}_${SPLIT}/best_model.pth"

    # Check if checkpoint exists
    if [ ! -f "$CHECKPOINT" ]; then
        echo ""
        echo "WARNING: Checkpoint not found: $CHECKPOINT"
        echo "Skipping $TASK..."
        continue
    fi

    echo ""
    echo "============================================================"
    echo "Evaluating: $TASK"
    echo "Checkpoint: $CHECKPOINT"
    echo "Mode: $MODE"
    echo "============================================================"

    # Run evaluation
    python src/eval/evaluate.py \
        --checkpoint "$CHECKPOINT" \
        --data_root . \
        --task "$TASK" \
        --split "$SPLIT" \
        --mode "$MODE" \
        --output_dir "$EVAL_OUTPUT_DIR"

    echo ""
    echo "✓ Completed evaluation for $TASK"
done

echo ""
echo "============================================================"
echo "All evaluations complete!"
echo "============================================================"
echo ""
echo "Results saved to:"
for TASK in "${TASKS[@]}"; do
    echo "  - $EVAL_OUTPUT_DIR/${TASK}_${MODE}_${SPLIT}_results.txt"
done
echo ""
echo "To view a summary:"
echo "  cat $EVAL_OUTPUT_DIR/*.txt"
