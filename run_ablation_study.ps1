# Activate conda environment
& C:\Users\Windows\miniconda3\shell\condabin\conda-hook.ps1
conda activate surgical_gestures_venv

$env:PYTHONPATH = "src"
$tasks = @("Suturing", "Needle_Passing", "Knot_Tying")
$configs = @(
    @{name="baseline";      yaml="baseline.yaml"},
    @{name="brain_eye";     yaml="brain_eye.yaml"},
    @{name="bridge_eeg";    yaml="bridge_eeg_rdm.yaml"},
    @{name="bridge_joint";  yaml="bridge_joint_eye_eeg.yaml"}
)

foreach ($cfg in $configs) {
    Write-Host "===== Starting condition: $($cfg.name) =====" -ForegroundColor Green
    foreach ($task in $tasks) {
        foreach ($fold in 1..8) {
            Write-Host "  $($cfg.name) | $task | fold_$fold"
            python src/training/train_vit_system.py `
                --config "src/configs/$($cfg.yaml)" `
                --task $task `
                --split "fold_$fold" `
                --data_root . `
                --output_dir "checkpoints/$($cfg.name)/$task/fold_$fold"
        }
    }
}

Write-Host "===== All training complete. Aggregating results... =====" -ForegroundColor Green
python aggregate_louo_results.py
