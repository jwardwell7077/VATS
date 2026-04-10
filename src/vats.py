"""
V.A.T.S. — Visual Aerial Targeting System

A real-time computer vision sensor for drone detection and tracking.
Outputs structured telemetry consumable by downstream fire control systems.

Pipeline stages:
    1. INGEST    — OpenCV grabs frames from webcam
    2. DETECT    — YOLOv8 finds drone bounding boxes
    3. TRACK     — ByteTrack assigns persistent IDs across frames
    4. PREDICT   — Kinematic buffer projects future position
    5. TELEMETRY — Structured output to stdout + visual overlay

This module is a *sensor*. It does not make engagement decisions.
A downstream system fuses V.A.T.S. output with RF data and rules of
engagement to decide whether to fire.
"""

import argparse
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Anchor relative paths to repo root so script runs from any CWD
os.chdir(Path(__file__).resolve().parents[1])


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Target:
    """A single tracked target at a single point in time."""
    track_id: int
    confidence: float
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2
    center: tuple[int, int]            # cx, cy

    # Prediction (None if insufficient history)
    predicted: tuple[int, int] | None = None
    speed_px: float = 0.0              # pixels per frame
    heading_rad: float = 0.0           # direction of travel
    cone_spread_rad: float = 0.0       # uncertainty fan width


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1: INGEST
# ═══════════════════════════════════════════════════════════════════════════

class VideoIngest:
    """Grabs frames from a webcam or video file.

    Forces MJPEG mode on USB webcams to avoid bandwidth issues over usbipd.
    """

    def __init__(self, source: int | str, width: int = 640, height: int = 480):
        self.source = source
        self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2 if isinstance(source, int) else 0)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        # Webcam tuning (MJPEG = lower bandwidth, critical over usbipd)
        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always grab newest frame

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self) -> np.ndarray | None:
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()


# ═══════════════════════════════════════════════════════════════════════════
# STAGES 2 & 3: DETECT + TRACK
# ═══════════════════════════════════════════════════════════════════════════

class DetectorTracker:
    """YOLOv8 detection + ByteTrack tracking, wrapped in one interface.

    Ultralytics fuses these into a single .track() call. Persistent IDs
    survive occlusions and dropped frames via Kalman filter prediction.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def process(self, frame: np.ndarray) -> list[Target]:
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf_threshold,
            verbose=False,
        )

        targets: list[Target] = []
        if not results or results[0].boxes is None:
            return targets

        boxes = results[0].boxes
        for box in boxes:
            # ByteTrack assigns IDs only to detections it accepts as tracks
            if box.id is None:
                continue

            track_id = int(box.id[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            targets.append(Target(
                track_id=track_id,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
            ))

        return targets


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4: PREDICT
# ═══════════════════════════════════════════════════════════════════════════

class KinematicPredictor:
    """Maintains a rolling position buffer per target ID and projects
    forward using averaged velocity.

    The cone spread represents directional uncertainty — wider when the
    target is changing direction erratically, narrower when flying straight.
    """

    def __init__(self, buffer_size: int = 8, lookahead_frames: int = 12):
        self.buffer_size = buffer_size
        self.lookahead = lookahead_frames
        self._buffers: dict[int, deque] = {}

    def update(self, targets: list[Target]):
        """Push current positions into per-target buffers and compute predictions."""
        active_ids = set()

        for target in targets:
            tid = target.track_id
            active_ids.add(tid)

            if tid not in self._buffers:
                self._buffers[tid] = deque(maxlen=self.buffer_size)

            self._buffers[tid].append(target.center)
            self._fill_prediction(target)

        # Drop buffers for targets we no longer see
        stale = set(self._buffers.keys()) - active_ids
        for tid in stale:
            del self._buffers[tid]

    def _fill_prediction(self, target: Target):
        """Compute speed/heading/spread/predicted from the rolling buffer.
        Mutates the Target dataclass in place."""
        buf = self._buffers[target.track_id]
        if len(buf) < 3:
            return  # need at least 3 points for a stable velocity

        pts = np.array(buf, dtype=np.float64)
        deltas = np.diff(pts, axis=0)

        # Mean velocity vector
        vx, vy = deltas.mean(axis=0)
        speed = math.hypot(vx, vy)

        if speed < 0.5:
            return  # essentially stationary, no useful prediction

        # Direction of motion
        heading = math.atan2(vy, vx)

        # Uncertainty: stddev of per-frame heading angles → cone spread
        per_frame_angles = np.arctan2(deltas[:, 1], deltas[:, 0])
        angle_std = float(np.std(per_frame_angles))
        spread = max(math.radians(8.0), angle_std * 1.5)
        spread = min(spread, math.radians(60.0))

        # Project forward
        last_x, last_y = pts[-1]
        pred_x = int(last_x + vx * self.lookahead)
        pred_y = int(last_y + vy * self.lookahead)

        target.predicted = (pred_x, pred_y)
        target.speed_px = speed
        target.heading_rad = heading
        target.cone_spread_rad = spread

    def get_trail(self, track_id: int) -> list[tuple[int, int]]:
        return list(self._buffers.get(track_id, []))


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 5: TELEMETRY (output + overlay)
# ═══════════════════════════════════════════════════════════════════════════

# Colors (BGR)
CLR_BBOX = (0, 255, 0)
CLR_TRAIL = (255, 200, 0)
CLR_PRED = (0, 0, 255)
CLR_CONE = (0, 0, 200)
CLR_HUD_BG = (15, 15, 15)
CLR_HUD_TEXT = (0, 255, 0)


def format_telemetry_line(t: Target) -> str:
    """Single-line structured output for downstream consumption."""
    if t.predicted:
        pred_str = f"({t.predicted[0]:>4},{t.predicted[1]:>4})"
        spd_str = f"{t.speed_px:>5.1f}"
        heading_deg = math.degrees(t.heading_rad)
        head_str = f"{heading_deg:>6.1f}"
        cone_deg = math.degrees(t.cone_spread_rad)
        cone_str = f"{cone_deg:>4.1f}"
    else:
        pred_str = "(----,----)"
        spd_str = "  ---"
        head_str = "  ----"
        cone_str = "----"

    return (
        f"[TGT-{t.track_id:03d}] "
        f"CONF {t.confidence:.2f} | "
        f"POS ({t.center[0]:>4},{t.center[1]:>4}) | "
        f"PRED {pred_str} | "
        f"SPD {spd_str} px/f | "
        f"HDG {head_str}° | "
        f"CONE {cone_str}°"
    )


def draw_overlay(frame: np.ndarray, targets: list[Target],
                 predictor: KinematicPredictor,
                 fps: float, frame_num: int):
    """Draw all visual annotations on the frame."""
    h, w = frame.shape[:2]

    # ── HUD bar (top) ─────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 40), CLR_HUD_BG, -1)
    cv2.putText(frame, "V.A.T.S. ONLINE", (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_HUD_TEXT, 2, cv2.LINE_AA)

    stats = f"FPS {fps:5.1f}  |  TGT {len(targets)}  |  FRM {frame_num}"
    tw = cv2.getTextSize(stats, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
    cv2.putText(frame, stats, (w - tw - 10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_HUD_TEXT, 1, cv2.LINE_AA)

    # ── Per-target overlays ───────────────────────────────────────────
    for t in targets:
        x1, y1, x2, y2 = t.bbox
        cx, cy = t.center

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), CLR_BBOX, 2)

        # Label
        label = f"TGT-{t.track_id:03d}  {t.confidence:.0%}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLR_BBOX, 2, cv2.LINE_AA)

        # Trail
        trail = predictor.get_trail(t.track_id)
        for i, (tx, ty) in enumerate(trail):
            alpha = (i + 1) / len(trail)
            color = tuple(int(c * alpha) for c in CLR_TRAIL)
            cv2.circle(frame, (tx, ty), max(1, int(3 * alpha)), color, -1)

        # Prediction marker + line
        if t.predicted:
            px, py = t.predicted
            cv2.line(frame, (cx, cy), (px, py), CLR_PRED, 1, cv2.LINE_AA)
            # Crosshair at predicted point
            cv2.line(frame, (px - 12, py), (px + 12, py), CLR_PRED, 2, cv2.LINE_AA)
            cv2.line(frame, (px, py - 12), (px, py + 12), CLR_PRED, 2, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 6, CLR_PRED, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run(source: str, model_path: str, conf: float, show: bool,
        save_path: str | None):

    # Parse source: int → webcam index, otherwise file path
    src: int | str = int(source) if source.isdigit() else source

    print(f"[VATS] Loading model: {model_path}")
    detector = DetectorTracker(model_path, conf_threshold=conf)

    print(f"[VATS] Opening source: {source}")
    ingest = VideoIngest(src)
    print(f"[VATS] Stream: {ingest.width}x{ingest.height}")

    predictor = KinematicPredictor()

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, 30.0,
                                  (ingest.width, ingest.height))
        print(f"[VATS] Recording to: {save_path}")

    print()
    print("=" * 90)
    print("   V.A.T.S. — TELEMETRY FEED")
    print("=" * 90)

    frame_num = 0
    fps_smooth = 0.0

    try:
        while True:
            t0 = time.perf_counter()

            frame = ingest.read()
            if frame is None:
                print("[VATS] End of stream")
                break
            frame_num += 1

            # ── Pipeline stages ─────────────────────────────────────────
            targets = detector.process(frame)        # Stages 2+3
            predictor.update(targets)                 # Stage 4

            # ── Telemetry output (Stage 5) ──────────────────────────────
            for t in targets:
                print(f"   {format_telemetry_line(t)}")

            # ── FPS measurement ─────────────────────────────────────────
            elapsed = time.perf_counter() - t0
            fps_smooth = 0.9 * fps_smooth + 0.1 / max(elapsed, 1e-6)

            # ── Visualization ───────────────────────────────────────────
            if show or writer:
                draw_overlay(frame, targets, predictor, fps_smooth, frame_num)

            if writer:
                writer.write(frame)

            if show:
                cv2.imshow("V.A.T.S.", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("\n[VATS] Operator terminated")
                    break

    finally:
        ingest.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        print("=" * 90)
        print(f"   SESSION END — {frame_num} frames")
        print("=" * 90)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="V.A.T.S. — Visual Aerial Targeting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 src/vats.py 0                          # webcam\n"
            "  python3 src/vats.py video.mp4                  # video file\n"
            "  python3 src/vats.py 0 -o out.mp4               # save annotated\n"
            "  python3 src/vats.py 0 --no-show                # headless telemetry\n"
        ),
    )
    p.add_argument("source", help="Webcam index (0, 1...) or video file path")
    p.add_argument("-m", "--model", default="models/vats_drone.pt",
                   help="YOLO model path (default: models/vats_drone.pt)")
    p.add_argument("-c", "--conf", type=float, default=0.25,
                   help="Detection confidence floor (default: 0.25)")
    p.add_argument("-o", "--output", default=None,
                   help="Save annotated video to this file")
    p.add_argument("--no-show", action="store_true",
                   help="Headless mode (telemetry only, no GUI)")

    args = p.parse_args()

    try:
        run(
            source=args.source,
            model_path=args.model,
            conf=args.conf,
            show=not args.no_show,
            save_path=args.output,
        )
    except RuntimeError as e:
        print(f"[VATS] FATAL: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[VATS] Interrupted")


if __name__ == "__main__":
    main()
