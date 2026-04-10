"""
V.A.T.S. — Label Review Tool

Step through pseudo-labeled frames and delete the bad ones.
The auto-labeler isn't perfect — your job is to remove frames where
the box is wrong (e.g. the model boxed your face instead of the drone).

Reads:    dataset/frames/*.jpg + dataset/labels/*.txt + .conf sidecars

Boxes are color-coded by auto-labeler confidence:
    green   conf >= 0.70   (high)
    yellow  conf >= 0.50   (medium)
    red     conf <  0.50   (low — likely junk)

Default behavior shows only frames that have at least one box. Frames
with empty labels (the model said "no drone here") are kept as
background negatives without review. Pass --all to review every frame.

Controls:
    SPACE / D = delete this frame + label, advance
    K         = keep this frame, advance
    A         = previous frame
    Q / ESC   = quit

Usage:
    python3 review.py
    python3 review.py --all
"""

import argparse
import os
from pathlib import Path

import cv2

# Anchor relative paths to repo root so script runs from any CWD
os.chdir(Path(__file__).resolve().parents[1])

FRAMES_DIR = Path("dataset/frames")
LABELS_DIR = Path("dataset/labels")


def load_boxes(label_path: Path, img_w: int, img_h: int):
    """Read YOLO label + .conf sidecar → list of (x1,y1,x2,y2,conf)."""
    if not label_path.exists():
        return []

    conf_path = label_path.with_suffix(".conf")
    confs = []
    if conf_path.exists():
        for line in conf_path.read_text().strip().splitlines():
            try:
                confs.append(float(line))
            except ValueError:
                confs.append(None)

    boxes = []
    for i, line in enumerate(label_path.read_text().strip().splitlines()):
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, w, h = map(float, parts)
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        c = confs[i] if i < len(confs) else None
        boxes.append((x1, y1, x2, y2, c))
    return boxes


def conf_color(c):
    """BGR color for a box based on its confidence tier."""
    if c is None:
        return (0, 255, 0)            # green (no conf info)
    if c >= 0.70:
        return (0, 255, 0)            # green: high
    if c >= 0.50:
        return (0, 220, 255)          # yellow: medium
    return (0, 80, 255)               # red: low


def draw(img, boxes, idx, total, status):
    h, w = img.shape[:2]
    for x1, y1, x2, y2, c in boxes:
        color = conf_color(c)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"drone {c:.2f}" if c is not None else "drone"
        cv2.putText(img, label, (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    cv2.rectangle(img, (0, 0), (w, 36), (15, 15, 15), -1)
    info = f"{idx + 1}/{total}  boxes:{len(boxes)}  {status}"
    cv2.putText(img, info, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    hint = "SPACE/D=del  K=keep  A=prev  Q=quit  |  green>=.70  yellow>=.50  red<.50"
    cv2.rectangle(img, (0, h - 28), (w, h), (15, 15, 15), -1)
    cv2.putText(img, hint, (10, h - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def select_frames(frames_dir: Path, labels_dir: Path, show_all: bool):
    images = sorted(frames_dir.glob("*.jpg"))
    if show_all:
        return images
    keep = []
    for img in images:
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists() and lbl.read_text().strip():
            keep.append(img)
    return keep


def main():
    parser = argparse.ArgumentParser(description="Review pseudo-labeled frames")
    parser.add_argument("--frames-dir", default=str(FRAMES_DIR))
    parser.add_argument("--labels-dir", default=str(LABELS_DIR))
    parser.add_argument("--all", action="store_true",
                        help="Review every frame, including empty-label backgrounds")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    labels_dir = Path(args.labels_dir)

    images = select_frames(frames_dir, labels_dir, args.all)
    if not images:
        print(f"[REVIEW] No frames to review in {frames_dir}/")
        return

    total_on_disk = len(list(frames_dir.glob("*.jpg")))
    print(f"[REVIEW] {len(images)} frame(s) selected for review "
          f"({total_on_disk} on disk total)")
    print("[REVIEW] SPACE/D = delete, K = keep, A = back, Q = quit")
    print()

    idx = 0
    deleted = 0
    last_status = ""
    while 0 <= idx < len(images):
        img_path = images[idx]
        label_path = labels_dir / f"{img_path.stem}.txt"
        conf_path = label_path.with_suffix(".conf")

        img = cv2.imread(str(img_path))
        if img is None:
            idx += 1
            continue

        h, w = img.shape[:2]
        boxes = load_boxes(label_path, w, h)
        display = draw(img.copy(), boxes, idx, len(images), last_status)

        cv2.imshow("V.A.T.S. Review", display)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("a"):
            idx = max(0, idx - 1)
            last_status = ""
        elif key in (ord(" "), ord("d")):
            try:
                img_path.unlink()
                if label_path.exists():
                    label_path.unlink()
                if conf_path.exists():
                    conf_path.unlink()
                deleted += 1
                last_status = f"DELETED ({deleted})"
            except OSError as e:
                last_status = f"ERR {e}"
            # Rebuild list after delete; idx now points to the next item
            images = select_frames(frames_dir, labels_dir, args.all)
            if idx >= len(images):
                break
        elif key == ord("k"):
            idx += 1
            last_status = "KEPT"

    cv2.destroyAllWindows()
    remaining = len(list(frames_dir.glob("*.jpg")))
    print()
    print(f"[REVIEW] Done.")
    print(f"  Deleted:    {deleted}")
    print(f"  Remaining:  {remaining}")
    print()
    print("[REVIEW] Next: python3 prepare_dataset.py")


if __name__ == "__main__":
    main()
