#!/bin/bash
# Script to extract per-class detection metrics for AR, MR, PR, TR
# Run this in TWCC environment

cd yolov5c

echo "=========================================="
echo "Extracting Per-Class Detection Metrics"
echo "=========================================="

# Models to validate
models=(
    "yolov5sc_backbone"
    "yolov5sc_p3"
    "yolov5sc_p4"
    "yolov5sc_p5"
    "yolov5mc_backbone"
    "yolov5mc_p3"
    "yolov5mc_p4"
    "yolov5mc_p5"
    "yolov5mlc_backbone"
    "yolov5mlc_p3"
    "yolov5mlc_p4"
    "yolov5mlc_p5"
)

versions=("v1" "v2" "v3" "v4" "v5")

total=$((${#models[@]} * ${#versions[@]}))
current=0

for model in "${models[@]}"; do
    for version in "${versions[@]}"; do
        current=$((current + 1))
        
        # Extract dataset number (v1 -> 1)
        dataset_num="${version#v}"
        
        weight_path="thesis results/${model}_${version}/weights/last.pt"
        data_yaml="../Regurgitation-YOLODataset-${dataset_num}/data.yaml"
        output_name="${model}_${version}_perclass"
        
        echo ""
        echo "[$current/$total] Validating: $output_name"
        
        if [ -f "$weight_path" ]; then
            python val.py \
                --weights "$weight_path" \
                --data "$data_yaml" \
                --batch-size 32 \
                --img 416 \
                --task test \
                --save-txt \
                --save-conf \
                --verbose \
                --project runs/val_perclass \
                --name "$output_name" \
                --exist-ok \
                2>&1 | tee "runs/val_perclass/${output_name}_log.txt"
            
            echo "  [OK] Completed: $output_name"
        else
            echo "  [WARN] Weight not found: $weight_path"
        fi
    done
done

echo ""
echo "=========================================="
echo "Extraction Complete!"
echo "=========================================="
echo "Results saved in: yolov5c/runs/val_perclass/"
echo ""
echo "Next step: Parse the validation logs to extract per-class mAP for AR, MR, PR, TR"



