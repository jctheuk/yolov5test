# Convert detection+classification dataset to YOLOv5 classification folder format
#
# Usage (PowerShell):
#   pwsh -File .\create_yolov5_classification_dataset.ps1
# or:
#   powershell -ExecutionPolicy Bypass -File .\create_yolov5_classification_dataset.ps1
#
# This script reads labels from regurgitationV1/<split>/labels/*.txt where each label file contains:
#   Line 1: detection bbox (ignored here)
#   Line 2: "<det_cls> <cls_idx> <present_flag>"  (we only use <cls_idx>)
# It copies the corresponding image from regurgitationV1/<split>/images to
#   <OutputRoot>/<split>/<ClassName>/<image_file>

param(
    [string]$SourceDatasetRoot = "regurgitationV1",
    [string]$OutputRoot = "regurgitationV1_classify",
    [string[]]$ClsNames = @("A4C", "PSAX", "PLAX"),
    [string[]]$ImageExtPriority = @(".png", ".jpg", ".jpeg", ".bmp")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Ensure-ClassifyDirs {
    param(
        [string]$BaseDir,
        [string[]]$Classes
    )
    foreach ($split in @('train','valid','test')) {
        foreach ($cls in $Classes) {
            $dir = Join-Path (Join-Path $BaseDir $split) $cls
            if (-not (Test-Path -LiteralPath $dir)) {
                New-Item -ItemType Directory -Path $dir | Out-Null
            }
        }
    }
}

function Resolve-ImagePath {
    param(
        [string]$LabelsDir,
        [string]$ImagesDir,
        [System.IO.FileInfo]$LabelFile,
        [string[]]$ExtPriority
    )
    # Compute candidate image stem relative to images dir
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($LabelFile.Name)
    foreach ($ext in $ExtPriority) {
        $candidate = Join-Path $ImagesDir ($stem + $ext)
        if (Test-Path -LiteralPath $candidate) {
            return (Get-Item -LiteralPath $candidate)
        }
    }
    return $null
}

function Parse-ClassificationIndex {
    param(
        [string[]]$LabelLines
    )
    # Support one-hot on second line, e.g. "1 0 0" or "100" or "0 0 1"
    $nonEmpty = $LabelLines | Where-Object { $_ -ne $null -and $_.Trim().Length -gt 0 }
    if ($nonEmpty.Count -lt 2) { return $null }
    $line = $nonEmpty[1].Trim()

    # Extract only 0/1 characters from the line
    $bitChars = @()
    foreach ($ch in $line.ToCharArray()) {
        if ($ch -eq '0' -or $ch -eq '1') { $bitChars += $ch }
    }
    # If we have at least three bits, interpret first three as one-hot
    if ($bitChars.Count -ge 3) {
        $bits = @([int]$bitChars[0].ToString(), [int]$bitChars[1].ToString(), [int]$bitChars[2].ToString())
        $oneIdx = [Array]::IndexOf($bits, 1)
        if ($oneIdx -ge 0) { return [int]$oneIdx }
    }

    # Fallback: try space-separated tokens and treat as [det_cls, cls_idx, present]
    $tokens = $line.Split([char]' ', [System.StringSplitOptions]::RemoveEmptyEntries)
    if ($tokens.Count -ge 2) {
        $tmp = 0
        $ok = [int]::TryParse($tokens[1], [ref]$tmp)
        if ($ok) { return [int]$tmp }
    }
    return $null
}

Write-Host "[INFO] Source dataset root: $SourceDatasetRoot"
Write-Host "[INFO] Output root: $OutputRoot"
Write-Host "[INFO] Classes: $($ClsNames -join ', ')"

Ensure-ClassifyDirs -BaseDir $OutputRoot -Classes $ClsNames

$stats = @{}
foreach ($split in @('train','valid','test')) {
    $stats[$split] = [ordered]@{ Total = 0 }
    foreach ($c in 0..($ClsNames.Length-1)) { $stats[$split]["$c"] = 0 }

    $labelsDir = Join-Path (Join-Path $SourceDatasetRoot $split) 'labels'
    $imagesDir = Join-Path (Join-Path $SourceDatasetRoot $split) 'images'
    if (-not (Test-Path -LiteralPath $labelsDir)) {
        Write-Warning "[WARN] Labels directory not found: $labelsDir. Skipping $split."
        continue
    }
    $labelFiles = Get-ChildItem -LiteralPath $labelsDir -Filter '*.txt' -File -Recurse
    foreach ($lf in $labelFiles) {
        $stats[$split]['Total']++
        $lines = Get-Content -LiteralPath $lf.FullName -Raw -ErrorAction Stop -Encoding UTF8
        $linesArr = $lines -split "`n"
        $clsIdx = Parse-ClassificationIndex -LabelLines $linesArr
        if ($null -eq $clsIdx) {
            Write-Warning "[WARN] Cannot parse classification index in $($lf.FullName). Skipping."
            continue
        }
        if (($clsIdx -lt 0) -or ($clsIdx -ge $ClsNames.Length)) {
            Write-Warning "[WARN] Classification index $clsIdx out of range in $($lf.FullName). Skipping."
            continue
        }

        $img = Resolve-ImagePath -LabelsDir $labelsDir -ImagesDir $imagesDir -LabelFile $lf -ExtPriority $ImageExtPriority
        if ($null -eq $img) {
            Write-Warning "[WARN] Image not found for label $($lf.Name) under $imagesDir. Tried: $($ImageExtPriority -join ', ')."
            continue
        }

        $clsName = $ClsNames[$clsIdx]
        $destDir = Join-Path (Join-Path $OutputRoot $split) $clsName
        if (-not (Test-Path -LiteralPath $destDir)) { New-Item -ItemType Directory -Path $destDir | Out-Null }
        $destPath = Join-Path $destDir $img.Name

        Copy-Item -LiteralPath $img.FullName -Destination $destPath -Force
        $stats[$split]["$clsIdx"]++
    }
}

# Print stats
Write-Host "`n[RESULT] Copy summary by split and class:"
foreach ($split in @('train','valid','test')) {
    if (-not $stats.ContainsKey($split)) { continue }
    $total = $stats[$split]['Total']
    $perClass = ($stats[$split].Keys | Where-Object { $_ -match '^[0-9]+$' } | Sort-Object {[int]$_} | ForEach-Object { "$_=$($stats[$split][$_])" }) -join ', '
    Write-Host (" - {0}: Total={1}, {2}" -f $split, $total, $perClass)
}

# Write CSV summary
$csvRows = @()
foreach ($split in @('train','valid','test')) {
    if (-not $stats.ContainsKey($split)) { continue }
    foreach ($c in 0..($ClsNames.Length-1)) {
        $csvRows += [pscustomobject]@{
            split = $split
            class_index = $c
            class_name = $ClsNames[$c]
            count = $stats[$split]["$c"]
        }
    }
}
if ($csvRows.Count -gt 0) {
    $csvPath = Join-Path $OutputRoot 'summary.csv'
    $csvRows | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8
    Write-Host "[INFO] Wrote summary to $csvPath"
}

Write-Host "`n[INFO] Done. You can now train with YOLOv5 classify using:"
Write-Host "  python .\\yolov5orignal\\classify\\train.py --data $OutputRoot --model yolov5s-cls.pt --epochs 10 --img 224"


