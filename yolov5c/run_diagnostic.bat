@echo off
echo Starting YOLOv5WithClassification Diagnostic...
echo.

cd /d "%~dp0"

python debug_training.py

echo.
echo Diagnostic complete! Check diagnostic_results/ folder for results.
pause
