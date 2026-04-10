"""
V.A.T.S. — Auto-Label Tool

Uses the current best detection model to pre-label extracted frames.
Outputs YOLO format labels (one .txt per image) plus a .conf sidecar
file with the per-box confidence scores (used by review.py for
color-coded display).

Reads from:  dataset/frames/*.jpg
Writes to:   dataset/labels/*.txt   (YOLO format, used for training)
             dataset/labels/*.conf  (one float per line, for review)

YOLO label format (one box per line):
    class_id  cx_norm  cy_norm  w_norm  h_norm

The auto-labeler is imperfect — that's the whole reason we're
fine-tuning. After it runs, use review.py to delete the bad frames
before training.

Usage:
    python3 auto_label.py
    python3 auto_label.py --conf 0.7        # be stricter
    python3 auto_label.py -m models/other.pt
"""

import argparse
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

# Anchor relative paths to repo root so script runs from any CWD
os.chdir(Path(__file__).resolve().parents[1])

FRAMES_DIR = Path("dataset/frames")
LABELS_DIR = Path("dataset/labels")


def main():
    parser = argparse.ArgumentParser(description="Pseudo-label frames with existing model")
    parser.add_argument("-m", "--model", default="models/yolov8x_drone.pt",
                        help="Detection model used for pseudo-labels")
    parser.add_argument("-c", "--conf", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5, raise for stricter labels)")
    parser.add_argument("--frames-dir", default=str(FRAMES_DIR))
    parser.add_argument("--labels-dir", default=str(LABELS_DIR))
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    labels_dir = Path(args.labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(frames_dir.glob("*.jpg"))
    if not images:
        print(f"[LABEL] No images in {frames_dir}/")
        print("[LABEL] Run extract_frames.py first.")
        return

    print(f"[LABEL] Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"[LABEL] Found {len(images)} frames")
    print(f"[LABEL] Confidence threshold: {args.conf}")
    print()

    n_with_box = 0
    n_empty = 0
    n_total_boxes = 0
    n_high = 0   # conf >= 0.70
    n_mid = 0    # 0.50 - 0.70
    n_low = 0    # < 0.50

    for i, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{i}/{len(images)}] {img_path.name}: read failed")
            continue
        h, w = img.shape[:2]

        results = model.predict(img, conf=args.conf, verbose=False)
        boxes = results[0].boxes if results else None

        lines = []
        confs = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                # Single class (drone) — any source class collapses to 0
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                confs.append(conf)
                if conf >= 0.70:
                    n_high += 1
                elif conf >= 0.50:
                    n_mid += 1
                else:
                    n_low += 1

        label_path = labels_dir / f"{img_path.stem}.txt"
        conf_path = labels_dir / f"{img_path.stem}.conf"
        if lines:
            label_path.write_text("\n".join(lines) + "\n")
            # Sidecar with one float per line, matching the boxes above.
            # YOLO label format is fixed at 5 fields, so we keep conf separate.
            conf_path.write_text("\n".join(f"{c:.4f}" for c in confs) + "\n")
            n_with_box += 1
            n_total_boxes += len(lines)
        else:
            # Empty file = "confirmed background frame".
            # YOLO trains on these as negative examples.
            label_path.write_text("")
            n_empty += 1

        if i % 25 == 0 or i == len(images):
            top = max(confs) if confs else 0.0
            print(f"  [{i}/{len(images)}] {img_path.name}: {len(lines)} box(es)  top={top:.2f}")

    print()
    print(f"[LABEL] Done.")
    print(f"  Frames with detections:  {n_with_box}")
    print(f"  Frames with no boxes:    {n_empty}  (kept as negative examples)")
    print(f"  Total boxes labeled:     {n_total_boxes}")
    print(f"    high conf (>=0.70):   {n_high}")
    print(f"    med  conf (>=0.50):   {n_mid}")
    print(f"    low  conf (< 0.50):   {n_low}")
    print()
    print("[LABEL] IMPORTANT: review labels before training:")
    print("  - delete frames where the model boxed your face/hand instead of the drone")
    print("  - delete frames where the box is way off")
    print("  - boxes are color-coded in review.py: green high / yellow med / red low")
    print("  - run:  python3 review.py")


if __name__ == "__main__":
    main()
