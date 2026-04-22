# Quick ablation study: Suturing only, 4 folds, 10 epochs, batch_size 32
# Estimated runtime: ~9-10 hours (or ~5 hours if batch_size 32 fits in VRAM)

# Activate conda environment
& C:\Users\Windows\miniconda3\shell\condabin\conda-hook.ps1
conda activate surgical_gestures_venv

$env:PYTHONPATH = "src"
$task = "Suturing"
$folds = 1..4
$configs = @(
    @{name="baseline";      yaml="quick_baseline.yaml"},
    @{name="brain_eye";     yaml="quick_brain_eye.yaml"},
    @{name="bridge_eeg";    yaml="quick_bridge_eeg_rdm.yaml"},
    @{name="bridge_joint";  yaml="quick_bridge_joint_eye_eeg.yaml"}
)

foreach ($cfg in $configs) {
    Write-Host "===== Starting condition: $($cfg.name) =====" -ForegroundColor Green
    foreach ($fold in $folds) {
        Write-Host "  $($cfg.name) | $task | fold_$fold" -ForegroundColor Cyan
        python src/training/train_vit_system.py `
            --config "src/configs/$($cfg.yaml)" `
            --task $task `
            --split "fold_$fold" `
            --data_root . `
            --output_dir "checkpoints/$($cfg.name)/$task/fold_$fold"
    }
}

Write-Host "===== All training complete =====" -ForegroundColor Green
