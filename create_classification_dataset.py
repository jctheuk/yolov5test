"""
Build a folder-per-class classification dataset from a YOLO multi-task dataset.

Source dataset layout (detection + classification):
  <root>/{train,valid,test}/images/*.png
  <root>/{train,valid,test}/labels/*.txt

Each label file is expected to contain at least 2 lines:
  - line 1: detection labels (ignored here)
  - line 2: classification one-hot or id. Examples seen:
        "0 0 1"  -> class id 2
        "1 0 0"  -> class id 0
        "2"      -> class id 2 (optional support)

This script reads the classification id and copies the corresponding image
into <out_root>/{train,val,test}/{A4C,PSAX,PLAX}/filename.ext.

Run:
  python create_classification_dataset.py --src regurgitationV1 --out cls_regurgitationV1

Notes:
  - Non-destructive: images are copied, originals untouched.
  - Skips files with missing/invalid labels, printing a warning.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CLASS_ID_TO_NAME = {
    0: "A4C",
    1: "PSAX",
    2: "PLAX",
}


def parse_class_id_from_label(label_path: Path) -> int | None:
    """Extract classification class id from label file.

    Supports either a one-hot vector (e.g., "0 1 0") or a single integer line (e.g., "1").
    The classification line is expected at line index 1 (second line). Extra whitespace tolerated.
    Returns None if parsing fails.
    """
    try:
        text = label_path.read_text(encoding="utf-8").strip().splitlines()
        if len(text) < 2:
            return None
        cls_line = text[1].strip()
        # Try one-hot first
        parts = [p for p in cls_line.replace(",", " ").split() if p]
        if len(parts) > 1:
            # Find index of max value
            floats = []
            for p in parts:
                try:
                    floats.append(float(p))
                except ValueError:
                    return None
            if not floats:
                return None
            cls_id = int(max(range(len(floats)), key=lambda i: floats[i]))
            return cls_id
        # Else treat as single integer id
        return int(cls_line)
    except Exception:
        return None


def convert_split(src_root: Path, out_root: Path, split: str, out_split: str | None = None) -> None:
    images_dir = src_root / split / "images"
    labels_dir = src_root / split / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        print(f"[WARN] Missing images or labels for split '{split}' at {src_root}")
        return

    # Prepare output class folders
    out_split = out_split or split
    for cls_name in CLASS_ID_TO_NAME.values():
        (out_root / out_split / cls_name).mkdir(parents=True, exist_ok=True)

    # Iterate labels; map to image by stem
    print(f"Processing {split} split...")
    for label_path in labels_dir.glob("*.txt"):
        cls_id = parse_class_id_from_label(label_path)
        if cls_id is None or cls_id not in CLASS_ID_TO_NAME:
            print(f"[SKIP] {label_path} -> invalid classification line")
            continue
        cls_name = CLASS_ID_TO_NAME[cls_id]
        print(f"Processing {label_path} -> class {cls_id} ({cls_name})")

        # Find paired image (support common extensions)
        stem = label_path.stem
        src_img = None
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            p = images_dir / f"{stem}{ext}"
            if p.exists():
                src_img = p
                break
        if src_img is None:
            print(f"[SKIP] Missing image for {label_path}")
            continue

        dst_img = out_root / out_split / cls_name / src_img.name
        if not dst_img.exists():
            print(f"Copying {src_img} -> {dst_img}")
            shutil.copy2(src_img, dst_img)


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO multi-task dataset to folder-per-class classification dataset.")
    parser.add_argument("--src", type=str, default="regurgitationV1", help="Source dataset root")
    parser.add_argument("--out", type=str, default="cls_regurgitationV1", help="Output dataset root")
    args = parser.parse_args()

    src_root = Path(args.src)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "test"):
        # Map 'valid' -> 'val' for YOLOv5 classification expected naming
        out_split = "val" if split == "valid" else split
        convert_split(src_root, out_root, split, out_split)

    print(f"Done. Classification dataset at: {out_root}")


if __name__ == "__main__":
    main()


