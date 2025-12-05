# This markdown walks you through how to run LOUO splits. You can do the full 8 fold split or create individual splits.


1. generate_splits.py

  - Generates LOUO splits for all three tasks
  - Shows skill level (N/I/E) for each surgeon
  - Prints a nice summary table

  2. train_vit_system.py 

  - New --split argument: Specify which fold to use (e.g., --split fold_1)
  - Validation loop: Runs validation after each epoch when a split is specified
  - Best model tracking: Saves best_model.pth whenever validation loss improves
  - Final model: Always saves final_model.pth at end of training
  - Improved logging: Shows both train and val losses separately

  ---
  How to Use

  Step 1: Generate splits (run once)

  python generate_splits.py

  This creates data/splits/{task}_splits.json for all three tasks.

  Step 2: Train with proper splits

  # Knot Tying
  python src/training/train_vit_system.py \
      --config src/configs/baseline.yaml \
      --data_root . \
      --task Knot_Tying \
      --split fold_1 \
      --output_dir checkpoints/knot_tying_fold1

  # Needle Passing
  python src/training/train_vit_system.py \
      --config src/configs/baseline.yaml \
      --data_root . \
      --task Needle_Passing \
      --split fold_1 \
      --output_dir checkpoints/needle_passing_fold1

  # Suturing
  python src/training/train_vit_system.py \
      --config src/configs/baseline.yaml \
      --data_root . \
      --task Suturing \
      --split fold_1 \
      --output_dir checkpoints/suturing_fold1

  Step 3: Evaluate on test set

  python src/eval/evaluate.py \
      --checkpoint checkpoints/knot_tying_fold1/best_model.pth \
      --data_root . \
      --task Knot_Tying \
      --split fold_1 \
      --mode test


# run training scripts:
./train_all_tasks.sh

# evaluate after training:

python src/eval/evaluate.py \
      --checkpoint checkpoints/knot_tying_fold_1/best_model.pth \
      --data_root . --task Knot_Tying --split fold_1 --mode test

  python src/eval/evaluate.py \
      --checkpoint checkpoints/needle_passing_fold_1/best_model.pth \
      --data_root . --task Needle_Passing --split fold_1 --mode test

  python src/eval/evaluate.py \
      --checkpoint checkpoints/suturing_fold_1/best_model.pth \
      --data_root . --task Suturing --split fold_1 --mode test