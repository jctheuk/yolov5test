"""
Compare class distributions between original labels (one-hot on line 2) and
YOLOv5 classify directory counts.

Usage:
  python compare_classify_distribution.py \
    --labels-root regurgitationV1 \
    --classify-root regurgitationV1_classify \
    --classes A4C PSAX PLAX
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--labels-root", default="regurgitationV1")
    p.add_argument("--classify-root", default="regurgitationV1_classify")
    p.add_argument("--classes", nargs="+", default=["A4C", "PSAX", "PLAX"])
    return p.parse_args()


def read_one_hot_index(line: str) -> int | None:
    s = line.strip()
    bits = [ch for ch in s if ch in {"0", "1"}]
    if len(bits) >= 3:
        first_three = bits[:3]
        try:
            idx = first_three.index("1")
            return idx
        except ValueError:
            return None
    toks = [t for t in s.split() if t]
    if len(toks) >= 2 and toks[1].isdigit():
        return int(toks[1])
    return None


def count_from_labels(labels_dir: Path, num_classes: int) -> List[int]:
    counts = [0] * num_classes
    for lf in labels_dir.glob("*.txt"):
        try:
            with lf.open("r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            if len(lines) < 2:
                continue
            idx = read_one_hot_index(lines[1])
            if idx is None or idx < 0 or idx >= num_classes:
                continue
            counts[idx] += 1
        except Exception:
            continue
    return counts


def count_from_classify(split_dir: Path, classes: List[str]) -> List[int]:
    counts = [0] * len(classes)
    for i, cls in enumerate(classes):
        d = split_dir / cls
        if not d.exists():
            continue
        counts[i] = sum(1 for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
    return counts


def main() -> None:
    args = parse_args()
    classes = args.classes
    for split in ["train", "valid", "test"]:
        labels_dir = Path(args.labels_root) / split / "labels"
        classify_dir = Path(args.classify_root) / split
        label_counts = count_from_labels(labels_dir, len(classes))
        folder_counts = count_from_classify(classify_dir, classes)
        total_labels = sum(label_counts)
        total_folder = sum(folder_counts)
        print(f"\n[Split: {split}]")
        print(f"  Labels: {label_counts} (Total={total_labels})")
        print(f"  Folder: {folder_counts} (Total={total_folder})")
        match = label_counts == folder_counts
        print(f"  Match: {match}")
        if not match:
            print("  Mismatch detected. Investigate label parsing or file mapping.")


if __name__ == "__main__":
    main()

