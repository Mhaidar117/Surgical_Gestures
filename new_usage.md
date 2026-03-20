Baseline:
```
python3 src/training/train_vit_system.py --config src/configs/baseline.yaml --data_root . --task Knot_Tying --split fold_1 --output_dir checkpoints/kt_fold1
```
Brain alignment"
```
python3 src/training/train_vit_system.py --config src/configs/brain_eye.yaml --data_root . --task all --split fold_1 --output_dir checkpoints/brain_fold1
```

8-fold LOUO with Brain:
```
./run_8fold_louo_brain.sh
```

Ablation:
```
./scripts/run_ablations.sh all
```