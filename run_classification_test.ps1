# YOLOv5 Classification Test - 1 Epoch
# Create test dataset and run training

Write-Host "🔄 Creating YOLOv5 Classification Test Dataset..." -ForegroundColor Green

# Create directory structure
$outputPath = "yolov5original\datasets\regurgitationV1-cls"
$classes = @("A4C", "PSAX", "PLAX")
$splits = @("train", "valid")

foreach ($split in $splits) {
    foreach ($cls in $classes) {
        $dir = Join-Path $outputPath $split $cls
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created: $dir"
    }
}

# Copy a few test images
Write-Host "`n📁 Copying test images..." -ForegroundColor Yellow

$sourcePath = "regurgitationV1"
$copiedCount = 0

foreach ($split in $splits) {
    $sourceImages = Join-Path $sourcePath $split "images"
    $sourceLabels = Join-Path $sourcePath $split "labels"
    
    if (Test-Path $sourceImages) {
        $images = Get-ChildItem -Path $sourceImages -Filter "*.png" | Select-Object -First 3
        
        foreach ($img in $images) {
            $labelFile = Join-Path $sourceLabels ($img.BaseName + ".txt")
            
            if (Test-Path $labelFile) {
                $content = Get-Content $labelFile
                if ($content.Count -ge 2) {
                    $clsLabel = $content[1].Split()
                    if ($clsLabel.Count -eq 3) {
                        $clsIdx = [array]::IndexOf($clsLabel, "1")
                        if ($clsIdx -ge 0) {
                            $clsName = $classes[$clsIdx]
                            $destPath = Join-Path $outputPath $split $clsName $img.Name
                            Copy-Item $img.FullName $destPath
                            Write-Host "  Copied $($img.Name) to $clsName"
                            $copiedCount++
                        }
                    }
                }
            }
        }
    }
}

Write-Host "`n📊 Copied $copiedCount test images" -ForegroundColor Green

# Run training
Write-Host "`n🚀 Starting 1-epoch classification training..." -ForegroundColor Green

Set-Location yolov5original

python classify/train.py `
    --data datasets/regurgitationV1-cls `
    --model yolov5s-cls.pt `
    --epochs 1 `
    --batch-size 4 `
    --imgsz 224 `
    --device cpu `
    --name test_1epoch `
    --project runs/train-cls `
    --exist-ok

Write-Host "`n✅ Training completed!" -ForegroundColor Green
Set-Location ..
