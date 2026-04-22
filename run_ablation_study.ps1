param(
    [string[]]$Tasks = @("Suturing", "Needle_Passing", "Knot_Tying"),
    [int]$StartFold = 1,
    [int]$EndFold = 8,
    [string[]]$Conditions = @("baseline", "brain_eye", "bridge_eeg", "bridge_joint"),
    [ValidateSet("none", "best", "all")]
    [string]$RetainCheckpoints = "best",
    [string]$DataRoot = ".",
    [string]$CheckpointRoot = "checkpoints",
    [string]$EvalRoot = "eval_results",
    [string]$AnalysisRoot = "analysis",
    [string]$EnvName = "surgical_gestures_venv",
    [switch]$SkipCondaActivation,
    [switch]$SkipExport,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = $PSScriptRoot
Set-Location $repoRoot
$env:PYTHONPATH = "src"

$conditionTable = [ordered]@{
    baseline     = "src/configs/baseline.yaml"
    brain_eye    = "src/configs/brain_eye.yaml"
    bridge_eeg   = "src/configs/bridge_eeg_rdm.yaml"
    bridge_joint = "src/configs/bridge_joint_eye_eeg.yaml"
}

function Initialize-CondaEnvironment {
    param(
        [string]$TargetEnv,
        [switch]$SkipActivation
    )

    if ($SkipActivation) {
        Write-Host "Skipping conda activation as requested." -ForegroundColor Yellow
        return
    }

    $condaCmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($null -eq $condaCmd) {
        Write-Warning "Conda was not found on PATH. Continuing without auto-activation."
        return
    }

    $hookScript = & $condaCmd.Source "shell.powershell" "hook" | Out-String
    Invoke-Expression $hookScript
    conda activate $TargetEnv
}

function Ensure-Directory {
    param([string]$Path)

    New-Item -ItemType Directory -Path $Path -Force | Out-Null
}

function Invoke-LoggedCommand {
    param(
        [string]$Label,
        [string[]]$Command,
        [string]$LogPath,
        [switch]$AllowDryRun
    )

    Ensure-Directory (Split-Path -Parent $LogPath)
    $rendered = ($Command | ForEach-Object {
        if ($_ -match "\s") { '"' + $_ + '"' } else { $_ }
    }) -join " "

    Write-Host ""
    Write-Host $Label -ForegroundColor Cyan
    Write-Host $rendered -ForegroundColor DarkGray

    if ($DryRun -and $AllowDryRun) {
        "[DRY RUN] $rendered" | Tee-Object -FilePath $LogPath | Out-Null
        return
    }

    $exe = $Command[0]
    $args = @()
    if ($Command.Length -gt 1) {
        $args = $Command[1..($Command.Length - 1)]
    }

    $cmdRendered = ($Command | ForEach-Object {
        if ($_ -match '[\s"]') {
            '"' + ($_ -replace '"', '\"') + '"'
        } else {
            $_
        }
    }) -join " "

    cmd /d /c "$cmdRendered 2>&1" | Tee-Object -FilePath $LogPath
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Command failed: $Label"
    }
}

function Cleanup-CheckpointArtifacts {
    param(
        [string]$RunDir,
        [string]$RetentionMode
    )

    if (-not (Test-Path $RunDir)) {
        return
    }

    switch ($RetentionMode) {
        "all" {
            return
        }
        "best" {
            Get-ChildItem -Path $RunDir -Filter "*.pth" -File | Where-Object {
                $_.Name -ne "best_model.pth"
            } | Remove-Item -Force
        }
        "none" {
            Get-ChildItem -Path $RunDir -Filter "*.pth" -File | Remove-Item -Force
        }
    }
}

function Get-BrainMode {
    param([string]$ConfigPath)

    if (-not (Test-Path $ConfigPath)) {
        throw "Config file not found: $ConfigPath"
    }

    $line = Select-String -Path $ConfigPath -Pattern '^\s*brain_mode\s*:' |
        Select-Object -First 1
    if ($null -eq $line) {
        return "none"
    }

    $value = ($line.Line -split ':', 2)[1].Trim()
    $value = ($value -split '#', 2)[0].Trim()
    $value = $value.Trim("'`"")
    if ([string]::IsNullOrWhiteSpace($value)) {
        return "none"
    }
    return $value
}

function Get-LogTail {
    param(
        [string]$LogPath,
        [int]$TailLines = 40
    )

    if (-not (Test-Path $LogPath)) {
        return ""
    }

    return ((Get-Content -Path $LogPath -Tail $TailLines) -join "`n")
}

function Get-RunStatus {
    param(
        [string]$LogPath
    )

    $logText = Get-LogTail -LogPath $LogPath
    if ($logText -match "Split fold_\d+ not found") {
        return "skipped_missing_fold"
    }
    if ($logText -match "OutOfMemoryError|CUDA out of memory") {
        return "failed_oom"
    }
    if ($logText -match "No results found in ") {
        return "skipped_no_results"
    }
    return "failed_other"
}

function Write-StatusArtifacts {
    param(
        [string]$ConditionDir,
        [System.Collections.ArrayList]$Statuses
    )

    $jsonPath = Join-Path $ConditionDir "run_status.json"
    $csvPath = Join-Path $ConditionDir "failed_runs.csv"

    $Statuses | ConvertTo-Json -Depth 6 | Set-Content -Path $jsonPath -Encoding utf8

    $failedRows = $Statuses | Where-Object { $_.status -ne "completed" }
    if ($failedRows.Count -eq 0) {
        Set-Content -Path $csvPath -Value '"condition","task","fold","stage","status","error_message","log_path"' -Encoding utf8
        return
    }

    $failedRows |
        ForEach-Object { [pscustomobject]$_ } |
        Select-Object condition, task, fold, stage, status, error_message, log_path |
        Export-Csv -Path $csvPath -NoTypeInformation -Encoding utf8
}

function Find-StatusIndex {
    param(
        [System.Collections.ArrayList]$Statuses,
        [string]$Condition,
        [string]$Task,
        [string]$Fold,
        [string]$Stage
    )

    for ($i = 0; $i -lt $Statuses.Count; $i++) {
        $row = $Statuses[$i]
        if ($row.condition -eq $Condition -and $row.task -eq $Task -and $row.fold -eq $Fold -and $row.stage -eq $Stage) {
            return $i
        }
    }

    return -1
}

function Add-RunStatus {
    param(
        [System.Collections.ArrayList]$Statuses,
        [string]$ConditionDir,
        [string]$Condition,
        [string]$Task,
        [string]$Fold,
        [string]$Stage,
        [string]$Status,
        [string]$ErrorMessage,
        [string]$LogPath
    )

    $statusRow = [ordered]@{
        condition = $Condition
        task = $Task
        fold = $Fold
        stage = $Stage
        status = $Status
        error_message = $ErrorMessage
        log_path = $LogPath
        timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
    }

    $existingIndex = Find-StatusIndex -Statuses $Statuses -Condition $Condition -Task $Task -Fold $Fold -Stage $Stage
    if ($existingIndex -ge 0) {
        $Statuses[$existingIndex] = $statusRow
    } else {
        [void]$Statuses.Add($statusRow)
    }

    Write-StatusArtifacts -ConditionDir $ConditionDir -Statuses $Statuses
}

function Update-OverallProgress {
    param(
        [int]$CompletedSteps,
        [int]$TotalSteps,
        [string]$CurrentStep,
        [System.Diagnostics.Stopwatch]$Stopwatch,
        [switch]$MarkCompleted
    )

    if ($TotalSteps -le 0) {
        return
    }

    $elapsed = $Stopwatch.Elapsed
    $percent = [Math]::Min(100, [Math]::Floor(($CompletedSteps / $TotalSteps) * 100))
    $status = "$CompletedSteps / $TotalSteps steps"
    $etaText = "ETA: calculating..."

    if ($CompletedSteps -gt 0 -and $CompletedSteps -lt $TotalSteps) {
        $avgSeconds = $elapsed.TotalSeconds / $CompletedSteps
        $remaining = [TimeSpan]::FromSeconds([Math]::Max(0, ($TotalSteps - $CompletedSteps) * $avgSeconds))
        $etaText = "ETA: {0:hh\:mm\:ss}" -f $remaining
    } elseif ($CompletedSteps -ge $TotalSteps) {
        $etaText = "ETA: 00:00:00"
    }

    $currentOperation = "{0} | Elapsed: {1:hh\:mm\:ss} | {2}" -f $CurrentStep, $elapsed, $etaText

    if ($MarkCompleted) {
        Write-Progress -Id 1 -Activity "Full 8-Fold Ablation + Validation" -Status $status -CurrentOperation $currentOperation -Completed
        return
    }

    Write-Progress -Id 1 -Activity "Full 8-Fold Ablation + Validation" -Status $status -CurrentOperation $currentOperation -PercentComplete $percent
}

Initialize-CondaEnvironment -TargetEnv $EnvName -SkipActivation:$SkipCondaActivation

$selectedConditions = @()
foreach ($condition in $Conditions) {
    if (-not $conditionTable.Contains($condition)) {
        throw "Unknown condition '$condition'. Valid options: $($conditionTable.Keys -join ', ')"
    }
    $selectedConditions += $condition
}

if ($StartFold -lt 1 -or $EndFold -lt $StartFold) {
    throw "Invalid fold range: $StartFold to $EndFold"
}

$timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
$pythonExe = "python"
$foldList = @($StartFold..$EndFold)
$comparisonRoot = Join-Path $AnalysisRoot "comparisons"
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

# Pre-compute per-condition brain_mode and step count:
# - brain_mode=none (baseline): per-(task,fold) train+eval = tasks * folds * 2 steps
# - brain_mode in {eye, bridge}: per-fold multi-task train + 3 per-task evals = folds * (1 + tasks)
$conditionBrainMode = @{}
$totalSteps = 0
foreach ($condition in $selectedConditions) {
    $bm = Get-BrainMode -ConfigPath $conditionTable[$condition]
    $conditionBrainMode[$condition] = $bm
    if ($bm -in @('eye','bridge')) {
        $totalSteps += $foldList.Count * (1 + $Tasks.Count)
    } else {
        $totalSteps += $Tasks.Count * $foldList.Count * 2
    }
}
$totalSteps += $selectedConditions.Count  # aggregate step per condition
if (-not $SkipExport) {
    $totalSteps += 1
}
$completedSteps = 0

Ensure-Directory $CheckpointRoot
Ensure-Directory $EvalRoot
Ensure-Directory $AnalysisRoot
Ensure-Directory $comparisonRoot

Write-Host "============================================================" -ForegroundColor Green
Write-Host "Full 8-Fold Ablation + Validation Runner" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Tasks: $($Tasks -join ', ')"
Write-Host "Folds: $StartFold to $EndFold"
Write-Host "Conditions: $($selectedConditions -join ', ')"
Write-Host "Retain checkpoints: $RetainCheckpoints"
Write-Host "Dry run: $DryRun"
Write-Host "============================================================" -ForegroundColor Green

foreach ($condition in $selectedConditions) {
    $configPath = $conditionTable[$condition]
    $conditionAnalysisDir = Join-Path $AnalysisRoot $condition
    $conditionEvalDir = Join-Path $EvalRoot $condition
    $conditionLogDir = Join-Path $conditionAnalysisDir "logs"
    $metadataPath = Join-Path $conditionAnalysisDir "run_metadata.json"
    $conditionStatuses = [System.Collections.ArrayList]::new()

    Ensure-Directory $conditionAnalysisDir
    Ensure-Directory $conditionEvalDir
    Ensure-Directory $conditionLogDir

    $conditionMetadata = [ordered]@{
        condition = $condition
        config_path = $configPath
        tasks = $Tasks
        folds = $foldList
        timestamp = $timestamp
        data_root = $DataRoot
        checkpoint_root = $CheckpointRoot
        eval_root = $conditionEvalDir
        analysis_root = $conditionAnalysisDir
        retain_checkpoints = $RetainCheckpoints
        arm = "PSM2"
        dry_run = [bool]$DryRun
        runner_script = "run_ablation_study.ps1"
        train_command_template = "python src/training/train_vit_system.py --config <config> --task <task> --split fold_<N> --data_root <data_root> --output_dir <run_dir> --arm PSM2"
        eval_command_template = "python src/eval/evaluate.py --checkpoint <run_dir>/best_model.pth --data_root <data_root> --task <task> --split fold_<N> --mode test --output_dir <eval_dir> --arm PSM2"
    }
    $conditionMetadata | ConvertTo-Json -Depth 5 | Set-Content -Path $metadataPath -Encoding utf8

    Write-Host ""
    $brainMode = $conditionBrainMode[$condition]
    $multiTask = $brainMode -in @('eye','bridge')
    Write-Host "===== Starting condition: $condition (brain_mode=$brainMode, multi_task=$multiTask) =====" -ForegroundColor Green

    if ($multiTask) {
        # Multi-task: one training run per fold using --task all, then 3 per-task evals
        # from the same checkpoint. Produces per-task fold result files compatible with
        # aggregate_louo_results.py and manuscript_writer.py.
        foreach ($fold in $foldList) {
            $split = "fold_$fold"
            $runDir = Join-Path $CheckpointRoot (Join-Path $condition (Join-Path 'all' $split))
            $trainLog = Join-Path $conditionLogDir "all_${split}_train.log"
            $bestCheckpoint = Join-Path $runDir "best_model.pth"

            Ensure-Directory $runDir

            $trainCmd = @(
                $pythonExe,
                "src/training/train_vit_system.py",
                "--config", $configPath,
                "--task", "all",
                "--split", $split,
                "--data_root", $DataRoot,
                "--output_dir", $runDir,
                "--arm", "PSM2"
            )
            Update-OverallProgress -CompletedSteps $completedSteps -TotalSteps $totalSteps -CurrentStep "$condition | all | $split | train" -Stopwatch $stopwatch
            $trainSucceeded = $false
            try {
                Invoke-LoggedCommand -Label "$condition | all | $split | train" -Command $trainCmd -LogPath $trainLog -AllowDryRun
                $trainSucceeded = $true
                Add-RunStatus -Statuses $conditionStatuses -ConditionDir $conditionAnalysisDir -Condition $condition -Task "all" -Fold $split -Stage "train" -Status "completed" -ErrorMessage "" -LogPath $trainLog
            } catch {
                $status = Get-RunStatus -LogPath $trainLog
                $errorMessage = Get-LogTail -LogPath $trainLog -TailLines 20
                Add-RunStatus -Statuses $conditionStatuses -ConditionDir $conditionAnalysisDir -Condition $condition -Task "all" -Fold $split -Stage "train" -Status $status -ErrorMessage $errorMessage -LogPath $trainLog
            }
            $completedSteps += 1

            if (-not $trainSucceeded) {
                foreach ($task in $Tasks) {
                    $evalLog = Join-Path $conditionLogDir "${task}_${split}_eval.log"
                    Add-RunStatus -Statuses $conditionStatuses -ConditionDir $conditionAnalysisDir -Condition $condition -Task $task -Fold $split -Stage "eval" -Status "skipped_upstream_failure" -ErrorMessage "Skipped because multi-task training did not complete." -LogPath $evalLog
                    $completedSteps += 1
                }
                continue
            }

            if (-not $DryRun -and -not (Test-Path $bestCheckpoint)) {
                foreach ($task in $Tasks) {
                    $evalLog = Join-Path $conditionLogDir "${task}_${split}_eval.log"
                    Add-RunStatus -Statuses $conditionStatuses -ConditionDir $conditionAnalysisDir -Condition $condition -Task $task -Fold $split -Stage "eval" -Status "failed_other" -ErrorMessage "Expected checkpoint not found: $bestCheckpoint" -LogPath $evalLog
                    $completedSteps += 1
                }
                continue
            }

            $allEvalsSucceeded = $true
            foreach ($task in $Tasks) {
                $evalLog = Join-Path $conditionLogDir "${task}_${split}_eval.log"
                $evalReport = Join-Path $conditionEvalDir "${task}_test_fold_${fold}_results.txt"

                $evalCmd = @(
                    $pythonExe,
                    "src/eval/evaluate.py",
                    "--checkpoint", $bestCheckpoint,
                    "--data_root", $DataRoot,
                    "--task", $task,
                    "--split", $split,
                    "--mode", "test",
                    "--output_dir", $conditionEvalDir,
                    "--arm", "PSM2"
                )
                Update-OverallProgress -CompletedSteps $completedSteps -TotalSteps $totalSteps -CurrentStep "$condition | $task | $split | eval" -Stopwatch $stopwatch
                try {
                    Invoke-LoggedCommand -Label "$condition | $task | $split | eval" -Command $evalCmd -LogPath $evalLog -AllowDryRun
                    if (-not $DryRun -and -not (Test-Path $evalReport)) {
                        throw "Expected evaluation report not found: $evalReport"
                    }
                    Add-RunStatus -Statuses $conditionStatuses -ConditionDir $conditionAnalysisDir -Condition $condition -Task $task -Fold $split -Stage "eval" -Status "completed" -ErrorMessage "" -LogPath $evalLog
                } catch {
                    $allEvalsSucceeded = $false
                    $status = Get-RunStatus -LogPath $evalLog
                    $errorMessage = Get-LogTail -LogPath $evalLog -TailLines 20
                    if (-not $errorMessage) {
                        $errorMessage = $_.Exception.Message
                    }
                    Add-RunStatus -Statuses $conditionStatuses -ConditionDir $conditionAnalysisDir -Condition $condition -Task $task -Fold $split -Stage "eval" -Status $status -ErrorMessage $errorMessage -LogPath $evalLog
                }
                $completedSteps += 1
            }

            if ($allEvalsSucceeded -and -not $DryRun) {
                Cleanup-CheckpointArtifacts -RunDir $runDir -RetentionMode $RetainCheckpoints
            }
        }
    } else {
        foreach ($task in $Tasks) {
            foreach ($fold in $foldList) {
                $split = "fold_$fold"
                $runDir = Join-Path $CheckpointRoot (Join-Path $condition (Join-Path $task $split))
                $trainLog = Join-Path $conditionLogDir "${task}_${split}_train.log"
                $evalLog = Join-Path $conditionLogDir "${task}_${split}_eval.log"
                $bestCheckpoint = Join-Path $runDir "best_model.pth"
                $evalReport = Join-Path $conditionEvalDir "${task}_test_fold_${fold}_results.txt"

                Ensure-Directory $runDir

                $trainCmd = @(
                    $pythonExe,
                    "src/training/train_vit_system.py",
                    "--config", $configPath,
                    "--task", $task,
                    "--split", $split,
                    "--data_root", $DataRoot,
                    "--output_dir", $runDir,
                    "--arm", "PSM2"
                )
                Update-OverallProgress -CompletedSteps $completedSteps -TotalSteps $totalSteps -CurrentStep "$condition | $task | $split | train" -Stopwatch $stopwatch
                $trainSucceeded = $false
                try {
                    Invoke-LoggedCommand -Label "$condition | $task | $split | train" -Command $trainCmd -LogPath $trainLog -AllowDryRun
                    $trainSucceeded = $true
                    Add-RunStatus -Statuses $conditionStatuses -ConditionDir $conditionAnalysisDir -Condition $condition -Task $task -Fold $split -Stage "train" -Status "completed" -ErrorMessage "" -LogPath $trainLog
                } catch {
                    $status = Get-RunStatus -LogPath $trainLog
                    $errorMessage = Get-LogTail -LogPath $trainLog -TailLines 20
                    Add-RunStatus -Statuses $conditionStatuses -ConditionDir $conditionAnalysisDir -Condition $condition -Task $task -Fold $split -Stage "train" -Status $status -ErrorMessage $errorMessage -LogPath $trainLog
                }
                $completedSteps += 1

                if (-not $trainSucceeded) {
                    Add-RunStatus -Statuses $conditionStatuses -ConditionDir $conditionAnalysisDir -Condition $condition -Task $task -Fold $split -Stage "eval" -Status "skipped_upstream_failure" -ErrorMessage "Skipped because training did not complete." -LogPath $evalLog
                    $completedSteps += 1
                    continue
                }

                if (-not $DryRun -and -not (Test-Path $bestCheckpoint)) {
                    Add-RunStatus -Statuses $conditionStatuses -ConditionDir $conditionAnalysisDir -Condition $condition -Task $task -Fold $split -Stage "eval" -Status "failed_other" -ErrorMessage "Expected checkpoint not found: $bestCheckpoint" -LogPath $evalLog
                    $completedSteps += 1
                    continue
                }

                $evalCmd = @(
                    $pythonExe,
                    "src/eval/evaluate.py",
                    "--checkpoint", $bestCheckpoint,
                    "--data_root", $DataRoot,
                    "--task", $task,
                    "--split", $split,
                    "--mode", "test",
                    "--output_dir", $conditionEvalDir,
                    "--arm", "PSM2"
                )
                Update-OverallProgress -CompletedSteps $completedSteps -TotalSteps $totalSteps -CurrentStep "$condition | $task | $split | eval" -Stopwatch $stopwatch
                $evalSucceeded = $false
                try {
                    Invoke-LoggedCommand -Label "$condition | $task | $split | eval" -Command $evalCmd -LogPath $evalLog -AllowDryRun
                    if (-not $DryRun -and -not (Test-Path $evalReport)) {
                        throw "Expected evaluation report not found: $evalReport"
                    }
                    $evalSucceeded = $true
                    Add-RunStatus -Statuses $conditionStatuses -ConditionDir $conditionAnalysisDir -Condition $condition -Task $task -Fold $split -Stage "eval" -Status "completed" -ErrorMessage "" -LogPath $evalLog
                } catch {
                    $status = Get-RunStatus -LogPath $evalLog
                    $errorMessage = Get-LogTail -LogPath $evalLog -TailLines 20
                    if (-not $errorMessage) {
                        $errorMessage = $_.Exception.Message
                    }
                    Add-RunStatus -Statuses $conditionStatuses -ConditionDir $conditionAnalysisDir -Condition $condition -Task $task -Fold $split -Stage "eval" -Status $status -ErrorMessage $errorMessage -LogPath $evalLog
                }
                $completedSteps += 1

                if ($evalSucceeded -and -not $DryRun) {
                    Cleanup-CheckpointArtifacts -RunDir $runDir -RetentionMode $RetainCheckpoints
                }
            }
        }
    }

    $summaryPath = Join-Path $conditionAnalysisDir "louo_summary.txt"
    $jsonPath = Join-Path $conditionAnalysisDir "louo_results.json"
    $aggregateLog = Join-Path $conditionLogDir "aggregate.log"
    # Resolve the aggregator path robustly: prefer pipeline/, fall back to repo
    # root for older checkouts. Use an absolute Windows path to avoid cmd /c
    # token-splitting on forward slashes (which previously produced
    # "can't open file 'aggregate_louo_results.py'" at runtime).
    $aggregateScriptCandidates = @(
        (Join-Path $PSScriptRoot "pipeline\aggregate_louo_results.py"),
        (Join-Path $PSScriptRoot "aggregate_louo_results.py")
    )
    $aggregateScript = $aggregateScriptCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $aggregateScript) {
        throw "aggregate_louo_results.py not found under $PSScriptRoot (tried: $($aggregateScriptCandidates -join '; '))"
    }
    $aggregateCmd = @(
        $pythonExe,
        $aggregateScript,
        "--eval_dir", $conditionEvalDir,
        "--output", $summaryPath,
        "--json_output", $jsonPath
    )
    Update-OverallProgress -CompletedSteps $completedSteps -TotalSteps $totalSteps -CurrentStep "$condition | aggregate" -Stopwatch $stopwatch
    try {
        Invoke-LoggedCommand -Label "$condition | aggregate" -Command $aggregateCmd -LogPath $aggregateLog -AllowDryRun
        if (-not $DryRun -and -not (Test-Path $jsonPath)) {
            throw "Expected aggregate JSON not found: $jsonPath"
        }
        Add-RunStatus -Statuses $conditionStatuses -ConditionDir $conditionAnalysisDir -Condition $condition -Task "*" -Fold "*" -Stage "aggregate" -Status "completed" -ErrorMessage "" -LogPath $aggregateLog
    } catch {
        $status = Get-RunStatus -LogPath $aggregateLog
        $errorMessage = Get-LogTail -LogPath $aggregateLog -TailLines 20
        if (-not $errorMessage) {
            $errorMessage = $_.Exception.Message
        }
        Add-RunStatus -Statuses $conditionStatuses -ConditionDir $conditionAnalysisDir -Condition $condition -Task "*" -Fold "*" -Stage "aggregate" -Status $status -ErrorMessage $errorMessage -LogPath $aggregateLog
    }
    $completedSteps += 1
}

if (-not $SkipExport) {
    $exportLog = Join-Path $comparisonRoot "export.log"
    $exportCmd = @(
        $pythonExe,
        "pipeline/export_ablation_analysis.py",
        "--analysis_root", $AnalysisRoot,
        "--eval_root", $EvalRoot,
        "--runner_script", "run_ablation_study.ps1",
        "--retain_checkpoints", $RetainCheckpoints,
        "--conditions"
    ) + $selectedConditions

    Update-OverallProgress -CompletedSteps $completedSteps -TotalSteps $totalSteps -CurrentStep "comparison export" -Stopwatch $stopwatch
    Invoke-LoggedCommand -Label "comparison export" -Command $exportCmd -LogPath $exportLog -AllowDryRun
    $completedSteps += 1
}

Update-OverallProgress -CompletedSteps $completedSteps -TotalSteps $totalSteps -CurrentStep "complete" -Stopwatch $stopwatch -MarkCompleted

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Ablation pipeline complete." -ForegroundColor Green
Write-Host "Analysis root: $AnalysisRoot"
Write-Host "Eval root: $EvalRoot"
Write-Host "Checkpoint retention: $RetainCheckpoints"
Write-Host "============================================================" -ForegroundColor Green
