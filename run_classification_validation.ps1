# YOLOv5 Original Classification Validation Script
# Run validation on all trained models to generate confusion matrices

Write-Host "Starting validation for all YOLOv5 classification models..." -ForegroundColor Green

# Model configurations
$models = @("yolov5s-cls", "yolov5m-cls", "yolov5l-cls")
$versions = @("v1", "v2", "v3", "v4", "v5")
$modelSuffixes = @{
    "yolov5s-cls" = "classifys"
    "yolov5m-cls" = "classifym"
    "yolov5l-cls" = "classifyl"
}

# Change to yolov5original directory
Set-Location yolov5original

$totalTasks = $models.Count * $versions.Count
$currentTask = 0

foreach ($model in $models) {
    foreach ($version in $versions) {
        $currentTask++
        $suffix = $modelSuffixes[$model]
        $expName = "${suffix}_${version}"
        $weightPath = "runs/train-cls/${expName}/weights/last.pt"
        $dataPath = "../regurgitation${version.ToUpper()}-Classification"
        
        Write-Host "`n[$currentTask/$totalTasks] Validating: $expName" -ForegroundColor Cyan
        Write-Host "  Weights: $weightPath" -ForegroundColor Gray
        Write-Host "  Data: $dataPath" -ForegroundColor Gray
        
        if (Test-Path $weightPath) {
            # Run validation with compute-metrics flag
            python classify/val.py --weights $weightPath --data $dataPath --batch-size 32 --imgsz 416 --compute-metrics --project runs/val-cls --name $expName --exist-ok --verbose
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  ✓ Validation completed successfully" -ForegroundColor Green
            } else {
                Write-Host "  ✗ Validation failed" -ForegroundColor Red
            }
        } else {
            Write-Host "  ⚠ Weight file not found: $weightPath" -ForegroundColor Yellow
        }
    }
}

Set-Location ..

Write-Host "`n✓ All validations completed!" -ForegroundColor Green
Write-Host "Results saved to: yolov5original/runs/val-cls/" -ForegroundColor Cyan

