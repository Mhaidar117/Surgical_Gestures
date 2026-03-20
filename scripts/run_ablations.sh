#!/bin/bash
#
# Run ablation studies for brain alignment.
# Compares: no brain vs eye-RDM-aligned, lambda sweep, ViT layers, adapters vs full fine-tune.
#
# Usage:
#   ./scripts/run_ablations.sh [ablation_type]
#   ./scripts/run_ablations.sh all          # Run all ablations (default)
#   ./scripts/run_ablations.sh brain         # Brain vs no-brain only
#   ./scripts/run_ablations.sh lambda        # Lambda sweep only
#   ./scripts/run_ablations.sh finetune      # Adapters vs full fine-tune only
#

set -e

ABLATION=${1:-all}
DATA_ROOT="."
SPLIT="fold_1"
BASE_OUTPUT="checkpoints/ablations"
mkdir -p "$BASE_OUTPUT"

run_train() {
    local config=$1
    local suffix=$2
    local extra_args=${3:-}
    echo ""
    echo "=============================================="
    echo "Running: $suffix (config: $config)"
    echo "=============================================="
    python3 src/training/train_vit_system.py \
        --config "$config" \
        --data_root "$DATA_ROOT" \
        --task all \
        --split "$SPLIT" \
        --output_dir "$BASE_OUTPUT/$suffix" \
        --arm PSM2 $extra_args
}

case "$ABLATION" in
    brain)
        echo "Ablation: Brain alignment (no brain vs eye-RDM-aligned)"
        # No brain: single-task baseline
        python3 src/training/train_vit_system.py \
            --config "src/configs/baseline.yaml" \
            --data_root "$DATA_ROOT" \
            --task Knot_Tying \
            --split "$SPLIT" \
            --output_dir "$BASE_OUTPUT/no_brain" \
            --arm PSM2
        # Eye-RDM aligned: multi-task with brain
        run_train "src/configs/brain_eye.yaml" "eye_rdm"
        ;;
    lambda)
        echo "Ablation: Lambda sweep (0.001, 0.01, 0.1)"
        for LAMBDA in 0.001 0.01 0.1; do
            # Create temp config with overridden lambda
            TMP_CONFIG="/tmp/brain_eye_lambda_${LAMBDA}.yaml"
            sed "s/brain: 0.01/brain: $LAMBDA/" src/configs/brain_eye.yaml > "$TMP_CONFIG"
            run_train "$TMP_CONFIG" "lambda_${LAMBDA}"
            rm -f "$TMP_CONFIG"
        done
        ;;
    finetune)
        echo "Ablation: Adapters only vs full ViT fine-tune"
        run_train "src/configs/brain_eye.yaml" "adapters_only"
        # Full fine-tune: freeze_until 0, use_adapters false
        TMP_CONFIG="/tmp/brain_eye_full_finetune.yaml"
        sed -e 's/freeze_until: 6/freeze_until: 0/' \
            -e 's/use_adapters: true/use_adapters: false/' \
            src/configs/brain_eye.yaml > "$TMP_CONFIG"
        run_train "$TMP_CONFIG" "full_finetune"
        rm -f "$TMP_CONFIG"
        ;;
    layers)
        echo "Ablation: ViT layers (early, mid, late)"
        for LAYER in early mid late; do
            TMP_CONFIG="/tmp/brain_eye_layer_${LAYER}.yaml"
            sed "s/brain_layer: mid/brain_layer: $LAYER/" src/configs/brain_eye.yaml > "$TMP_CONFIG"
            run_train "$TMP_CONFIG" "layer_${LAYER}"
            rm -f "$TMP_CONFIG"
        done
        ;;
    all)
        echo "Running all ablations..."
        $0 brain
        $0 lambda
        $0 finetune
        $0 layers
        ;;
    *)
        echo "Unknown ablation: $ABLATION"
        echo "Usage: $0 [all|brain|lambda|finetune|layers]"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Ablation(s) complete. Results in $BASE_OUTPUT/"
echo "=============================================="
