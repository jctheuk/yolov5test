# Extract per-class metrics for all models
# This will run validation on all model weights and extract detailed per-class metrics

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "            Per-Class Metrics Extraction Tool" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "This script will:" -ForegroundColor Green
Write-Host "  1. Run validation on all 12 YOLOv5c models (v1-v5)" -ForegroundColor White
Write-Host "  2. Extract per-class detection metrics (A4C, PSAX)" -ForegroundColor White
Write-Host "  3. Extract per-class classification metrics (A4C, PSAX)" -ForegroundColor White
Write-Host "  4. Generate comprehensive CSV and JSON outputs" -ForegroundColor White
Write-Host ""

Write-Host "Estimated time: 30-60 minutes (depending on hardware)" -ForegroundColor Yellow
Write-Host ""

# Ask for confirmation
$confirmation = Read-Host "Do you want to proceed? (yes/no)"

if ($confirmation -ne 'yes' -and $confirmation -ne 'y') {
    Write-Host ""
    Write-Host "Extraction cancelled." -ForegroundColor Red
    Write-Host ""
    Write-Host "Alternative: Check available per-class data first:" -ForegroundColor Cyan
    Write-Host "  python check_available_perclass_data.py" -ForegroundColor White
    Write-Host ""
    exit 0
}

Write-Host ""
Write-Host "Starting extraction..." -ForegroundColor Green
Write-Host ""

# Run the extraction script
python extract_perclass_metrics.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "                    Extraction Complete!" -ForegroundColor Green
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Generated files:" -ForegroundColor Yellow
    Write-Host "  [OK] results/perclass_metrics_detailed.json" -ForegroundColor Green
    Write-Host "  [OK] results/perclass_detection_metrics.csv" -ForegroundColor Green
    Write-Host "  [OK] results/perclass_classification_metrics.csv" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Review results: python -c 'import pandas as pd; print(pd.read_csv(\"results/perclass_detection_metrics.csv\"))'" -ForegroundColor White
    Write-Host "  2. See guide: cat PERCLASS_METRICS_GUIDE.md" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "================================================================================" -ForegroundColor Red
    Write-Host "                    Extraction Failed" -ForegroundColor Red
    Write-Host "================================================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Error code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Check if model weights exist (best.pt or last.pt)" -ForegroundColor White
    Write-Host "  2. Verify dataset paths are correct" -ForegroundColor White
    Write-Host "  3. Ensure GPU has enough memory" -ForegroundColor White
    Write-Host "  4. Check Python dependencies are installed" -ForegroundColor White
    Write-Host ""
    Write-Host "For help, see: PERCLASS_METRICS_GUIDE.md" -ForegroundColor Cyan
    Write-Host ""
    exit $LASTEXITCODE
}


