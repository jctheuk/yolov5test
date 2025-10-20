# Compute metrics for all YOLOv5 classification models

Write-Host "Computing per-class metrics for all models..." -ForegroundColor Green

$models = @(
    @{size="s"; prefix="classifys"},
    @{size="m"; prefix="classifym"},
    @{size="l"; prefix="classifyl"}
)
$versions = @("v1", "v2", "v3", "v4", "v5")

$totalTasks = $models.Count * $versions.Count
$currentTask = 0
$results = @()

foreach ($model in $models) {
    foreach ($version in $versions) {
        $currentTask++
        $modelName = "$($model.prefix)_$version"
        $versionUpper = $version.ToUpper()
        $weightPath = "yolov5original/runs/train-cls/$modelName/weights/last.pt"
        $dataPath = "regurgitation$versionUpper-Classification"
        
        Write-Host "`n[$currentTask/$totalTasks] Processing: $modelName" -ForegroundColor Cyan
        
        if (Test-Path $weightPath) {
            python compute_classification_metrics.py --weights $weightPath --data $dataPath --name $modelName --output classification_metrics --device cpu
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  ✓ Completed: $modelName" -ForegroundColor Green
                $results += $modelName
            } else {
                Write-Host "  ✗ Failed: $modelName" -ForegroundColor Red
            }
        } else {
            Write-Host "  ⚠ Weight file not found: $weightPath" -ForegroundColor Yellow
        }
    }
}

Write-Host "`n================================================================================`n" -ForegroundColor Green
Write-Host "✓ Completed $($results.Count)/$totalTasks models" -ForegroundColor Green
Write-Host "Results saved in: classification_metrics/" -ForegroundColor Cyan
Write-Host "`n================================================================================" -ForegroundColor Green



