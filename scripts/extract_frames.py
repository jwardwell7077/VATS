"""
V.A.T.S. — Frame Extraction Tool

Pulls frames out of raw video clips for dataset labeling.
Samples every Nth frame to avoid near-duplicate training images.

Accepts any common video format — capture.py output (clip_NNN.mp4)
and phone videos (.mov, .MP4, etc.) all work. Just drop them in
dataset/raw_videos/ and run.

Reads from:  dataset/raw_videos/*.{mp4,mov,avi}
Writes to:   dataset/frames/<stem>_frame_NNNN.jpg

Usage:
    python3 extract_frames.py
    python3 extract_frames.py --stride 10     # for longer phone clips
"""

import argparse
import os
from pathlib import Path

import cv2

# Anchor relative paths to repo root so script runs from any CWD
os.chdir(Path(__file__).resolve().parents[1])

RAW_DIR = Path("dataset/raw_videos")
FRAMES_DIR = Path("dataset/frames")
VIDEO_EXTS = ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI")


def extract_clip(video_path: Path, out_dir: Path, stride: int) -> int:
    """Extract every Nth frame from one clip. Returns count saved."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: cannot open {video_path.name} (codec issue?)")
        return 0

    clip_stem = video_path.stem  # e.g. clip_000 or IMG_4523
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {video_path.name}: {width}x{height} @ {fps:.0f}fps, {total} frames")

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            out_name = f"{clip_stem}_frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(out_dir / out_name), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"    -> {saved} frames saved")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract frames from raw videos")
    parser.add_argument("--stride", type=int, default=5,
                        help="Sample every Nth frame (default: 5)")
    parser.add_argument("--raw-dir", default=str(RAW_DIR),
                        help="Directory of raw video clips")
    parser.add_argument("--out-dir", default=str(FRAMES_DIR),
                        help="Output directory for extracted frames")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accept any common video format. capture.py output is .mp4, phone
    # videos are usually .mov — both work.
    clips = sorted({p for ext in VIDEO_EXTS for p in raw_dir.glob(ext)})
    if not clips:
        print(f"[EXTRACT] No video files found in {raw_dir}/")
        print("[EXTRACT] Run capture.py or drop phone videos in there first.")
        return

    print(f"[EXTRACT] Found {len(clips)} clip(s) in {raw_dir}/")
    print(f"[EXTRACT] Sampling every {args.stride} frames")
    print(f"[EXTRACT] Writing to {out_dir}/")
    print()

    total_saved = 0
    for clip in clips:
        total_saved += extract_clip(clip, out_dir, args.stride)

    print()
    print(f"[EXTRACT] Done. {total_saved} frames extracted total.")
    print(f"[EXTRACT] Next: python3 auto_label.py")


if __name__ == "__main__":
    main()
