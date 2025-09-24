@echo off
REM Class-Weighted Classification Training
REM Addresses class imbalance with weighted loss

echo Starting Class-Weighted Classification Training...
echo Using class weights: [PLAX: 2.13, PSAX: 4.78, A4C: 3.11]

python train_with_class_weights.py ^
    --data ../regurgitationV1/data.yaml ^
    --weights yolov5s.pt ^
    --epochs 100 ^
    --batch-size 16 ^
    --imgsz 640 ^
    --hyp yolov5c/data/hyps/hyp.improved_classification.yaml ^
    --device auto ^
    --project yolov5c/runs/class_weighted ^
    --name exp ^
    --patience 0 ^
    --save-period 10

echo Training completed!
pause
