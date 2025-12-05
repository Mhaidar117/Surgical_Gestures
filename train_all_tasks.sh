#!/bin/bash
#
# Train models for all three JIGSAWS tasks with proper LOUO splits
#
# Usage:
#   ./train_all_tasks.sh              # Train all tasks with fold_1
#   ./train_all_tasks.sh fold_2       # Train all tasks with fold_2
#   ./train_all_tasks.sh fold_1 Knot_Tying  # Train only Knot_Tying with fold_1
#

set -e  # Exit on error

# Configuration
SPLIT=${1:-fold_1}
TASK_FILTER=${2:-all}
CONFIG="src/configs/baseline.yaml"
DATA_ROOT="."
BASE_OUTPUT_DIR="checkpoints"

echo "============================================================"
echo "JIGSAWS Training Pipeline"
echo "============================================================"
echo "Split: $SPLIT"
echo "Config: $CONFIG"
echo "Output base: $BASE_OUTPUT_DIR"
echo "============================================================"

# Step 1: Generate splits if they don't exist
if [ ! -f "data/splits/Knot_Tying_splits.json" ]; then
    echo ""
    echo "Generating LOUO splits..."
    python generate_splits.py
fi

# Define tasks
TASKS=("Knot_Tying" "Needle_Passing" "Suturing")

# Train each task
for TASK in "${TASKS[@]}"; do
    # Skip if task filter is set and doesn't match
    if [ "$TASK_FILTER" != "all" ] && [ "$TASK_FILTER" != "$TASK" ]; then
        continue
    fi

    # Create output directory name (lowercase, underscores)
    TASK_LOWER=$(echo "$TASK" | tr '[:upper:]' '[:lower:]')
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TASK_LOWER}_${SPLIT}"

    echo ""
    echo "============================================================"
    echo "Training: $TASK"
    echo "Split: $SPLIT"
    echo "Output: $OUTPUT_DIR"
    echo "============================================================"

    # Run training
    python src/training/train_vit_system.py \
        --config "$CONFIG" \
        --data_root "$DATA_ROOT" \
        --task "$TASK" \
        --split "$SPLIT" \
        --output_dir "$OUTPUT_DIR" \
        --arm PSM2

    echo ""
    echo "✓ Completed training for $TASK"
    echo "  Best model: $OUTPUT_DIR/best_model.pth"
    echo "  Final model: $OUTPUT_DIR/final_model.pth"
done

echo ""
echo "============================================================"
echo "All training complete!"
echo "============================================================"
echo ""
echo "To evaluate the models, run:"
echo ""
for TASK in "${TASKS[@]}"; do
    if [ "$TASK_FILTER" != "all" ] && [ "$TASK_FILTER" != "$TASK" ]; then
        continue
    fi
    TASK_LOWER=$(echo "$TASK" | tr '[:upper:]' '[:lower:]')
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TASK_LOWER}_${SPLIT}"
    echo "  python src/eval/evaluate.py \\"
    echo "      --checkpoint $OUTPUT_DIR/best_model.pth \\"
    echo "      --data_root . --task $TASK --split $SPLIT --mode test"
    echo ""
done
