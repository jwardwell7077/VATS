"""
Fine-tune YOLOv8 Nano on the custom V.A.T.S. drone dataset.

Defaults to fine-tuning yolov8n.pt on dataset/vats/data.yaml with
augmentations turned ON (mosaic, rotation, HSV) — the things the
public yolov8x_drone model lacked, and the reason it overfit to
isolated/centered drones and false-fired on faces.

After training, the best weights are copied to: models/vats_drone.pt

Usage:
    python3 scripts/train.py                       # default: 50 epochs, vats dataset
    python3 scripts/train.py --epochs 100          # more epochs for better results
    python3 scripts/train.py --imgsz 320           # smaller images for faster training
    python3 scripts/train.py --data path/to.yaml   # different dataset
"""

import argparse
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

# Anchor relative paths to repo root so script runs from any CWD
os.chdir(Path(__file__).resolve().parents[1])


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n for drone detection")
    parser.add_argument("--data", default="dataset/vats/data.yaml",
                        help="Path to dataset YAML")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Training image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16, lower if OOM)")
    parser.add_argument("--name", default="vats_drone",
                        help="Run name (default: vats_drone)")
    parser.add_argument("--base", default="models/yolov8n.pt",
                        help="Base model to fine-tune (default: models/yolov8n.pt)")
    args = parser.parse_args()

    # Verify dataset exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Dataset YAML not found: {data_path}")
        print("Did you run prepare_dataset.py first?")
        return

    print("=" * 60)
    print("V.A.T.S. Drone Detector — Fine-Tuning")
    print("=" * 60)
    print(f"  Base model:   {args.base}")
    print(f"  Dataset:      {args.data}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Image size:   {args.imgsz}")
    print(f"  Batch size:   {args.batch}")
    print("=" * 60)

    # Load base model (will auto-download yolov8n.pt if not present)
    model = YOLO(args.base)

    # Augmentations turned ON. The public yolov8x_drone model trained
    # with mosaic=0 and degrees=0, which is a big part of why it
    # overfit to isolated/centered drones and false-fired on faces.
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=0,           # GPU 0 (the 4070)
        patience=15,        # early stop if no improvement
        save=True,
        plots=True,
        verbose=True,

        mosaic=1.0,         # full mosaic augmentation
        degrees=15.0,       # rotation up to ±15°
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    # Copy best weights to models/ directory for easy use
    runs_dir = Path("runs") / "detect" / args.name
    best_pt = runs_dir / "weights" / "best.pt"

    if best_pt.exists():
        dest = Path("models") / "vats_drone.pt"
        dest.parent.mkdir(exist_ok=True)
        shutil.copy(best_pt, dest)
        print()
        print("=" * 60)
        print(f"  TRAINING COMPLETE")
        print(f"  Best weights:  {best_pt}")
        print(f"  Copied to:     {dest}")
        print(f"  Test with:     python3 src/vats.py 0 -m {dest}")
        print("=" * 60)
    else:
        print(f"WARNING: Could not find best weights at {best_pt}")


if __name__ == "__main__":
    main()
