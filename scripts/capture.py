"""
V.A.T.S. — Dataset Capture Tool

Records short video clips from the webcam for custom dataset building.
Press SPACE to start/stop a recording, Q to quit.

Each recording is saved to: dataset/raw_videos/clip_NNN.mp4

Usage:
    python3 capture.py
    python3 capture.py --cam 0 --width 1280 --height 720
"""

import argparse
import os
import time
from pathlib import Path

import cv2

# Anchor relative paths to repo root so script runs from any CWD
os.chdir(Path(__file__).resolve().parents[1])

RAW_DIR = Path("dataset/raw_videos")


def find_next_index(directory: Path) -> int:
    """Find the next available clip_NNN.mp4 filename."""
    directory.mkdir(parents=True, exist_ok=True)
    existing = sorted(directory.glob("clip_*.mp4"))
    if not existing:
        return 0
    last = existing[-1].stem  # clip_007
    return int(last.split("_")[1]) + 1


def draw_hud(frame, recording: bool, clip_idx: int, duration: float):
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 44), (15, 15, 15), -1)
    if recording:
        # Pulsing red dot
        pulse = int((time.time() * 3) % 2)
        if pulse:
            cv2.circle(frame, (25, 22), 10, (0, 0, 255), -1)
        cv2.putText(frame, f"REC  clip_{clip_idx:03d}.mp4  [{duration:5.1f}s]",
                    (45, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, f"READY  next: clip_{clip_idx:03d}.mp4",
                    (15, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 0), 2, cv2.LINE_AA)

    # Bottom hint bar
    cv2.rectangle(frame, (0, h - 30), (w, h), (15, 15, 15), -1)
    hint = "SPACE: start/stop recording   |   Q: quit"
    cv2.putText(frame, hint, (15, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="V.A.T.S. dataset capture")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=1920,
                        help="Frame width (default 1920 for 1080p training data)")
    parser.add_argument("--height", type=int, default=1080, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    args = parser.parse_args()

    print(f"[CAPTURE] Opening camera {args.cam}...")
    cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("[CAPTURE] ERROR: Cannot open webcam")
        return

    # Force MJPEG (needed over usbipd) and target resolution
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    print(f"[CAPTURE] Camera: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")

    clip_idx = find_next_index(RAW_DIR)
    print(f"[CAPTURE] Next clip index: {clip_idx:03d}")
    print("[CAPTURE] Press SPACE to start/stop recording, Q to quit")

    writer: cv2.VideoWriter | None = None
    recording = False
    rec_start = 0.0
    clips_saved = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[CAPTURE] Frame read failed")
                break

            duration = time.time() - rec_start if recording else 0.0

            if recording and writer is not None:
                writer.write(frame)

            display = frame.copy()
            draw_hud(display, recording, clip_idx, duration)

            cv2.imshow("V.A.T.S. Capture", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # Q or ESC
                break

            if key == ord(" "):  # SPACE toggles recording
                if not recording:
                    out_path = RAW_DIR / f"clip_{clip_idx:03d}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(out_path), fourcc, actual_fps,
                        (actual_w, actual_h)
                    )
                    if not writer.isOpened():
                        print(f"[CAPTURE] ERROR: Could not open writer for {out_path}")
                        writer = None
                        continue
                    rec_start = time.time()
                    recording = True
                    print(f"[CAPTURE] ● REC  {out_path}")
                else:
                    writer.release()
                    writer = None
                    recording = False
                    clips_saved += 1
                    print(f"[CAPTURE] ■ STOP ({duration:.1f}s)  saved clip_{clip_idx:03d}.mp4")
                    clip_idx += 1

    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[CAPTURE] Done. {clips_saved} new clip(s) saved to {RAW_DIR}/")
        total = len(list(RAW_DIR.glob("clip_*.mp4")))
        print(f"[CAPTURE] Total clips in dataset: {total}")


if __name__ == "__main__":
    main()
