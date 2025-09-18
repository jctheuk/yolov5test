# Quick verifier for YOLOv5 classification dataset directory structure.
#
# Usage:
#   python verify_classification_dataset.py --root regurgitationV1_classify --classes A4C PSAX PLAX
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="regurgitationV1_classify", help="Path to classification root")
    parser.add_argument("--classes", type=str, nargs="+", default=["A4C", "PSAX", "PLAX"], help="Class names in order")
    return parser.parse_args()


essential_exts = {".png", ".jpg", ".jpeg", ".bmp"}


def count_images(root: Path, classes: List[str]) -> None:
    splits = ["train", "valid", "test"]
    for split in splits:
        print(f"\n[CHECK] Split: {split}")
        total = 0
        class_counts: Counter[str] = Counter()
        split_dir = root / split
        if not split_dir.exists():
            print(f"  [WARN] Missing split dir: {split_dir}")
            continue
        for cls in classes:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                print(f"  [WARN] Missing class dir: {cls_dir}")
                continue
            cnt = sum(1 for p in cls_dir.iterdir() if p.suffix.lower() in essential_exts)
            class_counts[cls] = cnt
            total += cnt
        print(f"  Total images: {total}")
        for cls in classes:
            print(f"   - {cls}: {class_counts[cls]}")


if __name__ == "__main__":
    args = parse_args()
    count_images(Path(args.root), args.classes)

