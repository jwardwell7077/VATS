"""
V.A.T.S. — Dataset Prep Tool

Splits labeled frames into train/val and writes a YOLO data.yaml.

Reads:   dataset/frames/*.jpg + dataset/labels/*.txt
Writes:  dataset/vats/train/images/, dataset/vats/train/labels/
         dataset/vats/val/images/,   dataset/vats/val/labels/
         dataset/vats/data.yaml

Usage:
    python3 prepare_dataset.py
    python3 prepare_dataset.py --val 0.15
"""

import argparse
import os
import random
import shutil
from pathlib import Path

# Anchor relative paths to repo root so script runs from any CWD
os.chdir(Path(__file__).resolve().parents[1])


def main():
    parser = argparse.ArgumentParser(description="Build YOLO dataset structure")
    parser.add_argument("--frames-dir", default="dataset/frames")
    parser.add_argument("--labels-dir", default="dataset/labels")
    parser.add_argument("--out-dir", default="dataset/vats")
    parser.add_argument("--val", type=float, default=0.2,
                        help="Validation split fraction (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)

    images = sorted(frames_dir.glob("*.jpg"))
    paired = [(img, labels_dir / f"{img.stem}.txt")
              for img in images
              if (labels_dir / f"{img.stem}.txt").exists()]

    if not paired:
        print(f"[PREP] No labeled frames found.")
        print(f"[PREP] Did you run auto_label.py?")
        return

    print(f"[PREP] Found {len(paired)} labeled frames")

    random.seed(args.seed)
    random.shuffle(paired)

    n_val = max(1, int(len(paired) * args.val))
    val = paired[:n_val]
    train = paired[n_val:]

    print(f"[PREP] Split: {len(train)} train / {len(val)} val")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / split / "images").mkdir(parents=True)
        (out_dir / split / "labels").mkdir(parents=True)

    for split_name, items in (("train", train), ("val", val)):
        for img, lbl in items:
            shutil.copy(img, out_dir / split_name / "images" / img.name)
            shutil.copy(lbl, out_dir / split_name / "labels" / lbl.name)

    data_yaml = out_dir / "data.yaml"
    yaml_text = f"""# V.A.T.S. custom drone dataset
path: {out_dir.resolve()}
train: train/images
val: val/images

names:
  0: drone
"""
    data_yaml.write_text(yaml_text)
    print(f"[PREP] Wrote {data_yaml}")
    print()
    print(f"[PREP] Done. Now train with:")
    print(f"  python3 train.py --data {data_yaml}")


if __name__ == "__main__":
    main()
