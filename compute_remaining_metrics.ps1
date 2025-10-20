# Compute metrics for remaining classification models

$models = @(
    @{prefix="classifys"; versions=@("v2", "v3", "v5")},
    @{prefix="classifym"; versions=@("v1", "v2", "v3", "v5")},
    @{prefix="classifyl"; versions=@("v1", "v2", "v3", "v4", "v5")}
)

$completed = 0
$total = 0
foreach ($model in $models) {
    $total += $model.versions.Count
}

Write-Host "Computing metrics for $total remaining models...`n" -ForegroundColor Green

foreach ($model in $models) {
    foreach ($version in $model.versions) {
        $completed++
        $modelName = "$($model.prefix)_$version"
        $versionUpper = $version.ToUpper()
        $weightPath = "yolov5original/runs/train-cls/$modelName/weights/last.pt"
        $dataPath = "regurgitation$versionUpper-Classification"
        
        Write-Host "[$completed/$total] Processing: $modelName" -ForegroundColor Cyan
        
        if (Test-Path $weightPath) {
            python compute_classification_metrics.py --weights $weightPath --data $dataPath --name $modelName --output classification_metrics --device cpu 2>&1 | Out-Null
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  [OK] Completed: $modelName" -ForegroundColor Green
            } else {
                Write-Host "  [ERROR] Failed: $modelName" -ForegroundColor Red
            }
        } else {
            Write-Host "  [WARN] Weight file not found" -ForegroundColor Yellow
        }
    }
}

Write-Host "`n[OK] Completed $completed models" -ForegroundColor Green



