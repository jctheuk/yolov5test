# Run aggregation script to generate comparison results
# This script aggregates v1-v5 metrics for all models and generates:
# - results/combined_metrics.csv
# - results/combined_table.tex
# - files/1760423080004_compared@2x.jpg

Write-Host "Starting aggregation of thesis results..." -ForegroundColor Green
Write-Host ""

# Run the aggregation script
python aggregate_thesis_results.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=== Aggregation Complete ===" -ForegroundColor Green
    Write-Host ""
    Write-Host "Generated files:" -ForegroundColor Cyan
    Write-Host "  1. results/combined_metrics.csv" -ForegroundColor Yellow
    Write-Host "  2. results/combined_table.tex" -ForegroundColor Yellow
    Write-Host "  3. files/1760423080004_compared@2x.jpg" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "See COMPARISON_RESULTS_SUMMARY.md for detailed analysis." -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "Error: Aggregation failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}


