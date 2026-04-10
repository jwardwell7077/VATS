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

# Colors (BGR) — high-contrast palette for projector / live demo
CLR_BBOX        = (80, 255, 120)    # bright lime — primary target box
CLR_TRAIL       = (255, 200, 0)     # cyan-amber trail
CLR_PRED        = (60, 60, 255)     # bright red — prediction line + reticle
CLR_HUD_BG      = (12, 12, 12)      # near-black panel
CLR_HUD_BORDER  = (60, 200, 60)     # green border accent
CLR_HUD_TEXT    = (180, 255, 180)   # soft green text
CLR_HUD_BRIGHT  = (80, 255, 120)    # title / values
CLR_HUD_DIM     = (110, 140, 110)   # secondary labels
CLR_STATUS_OK   = (80, 255, 120)
CLR_STATUS_WARN = (40, 220, 255)
CLR_STATUS_HOT  = (60, 60, 255)
CLR_STATUS_IDLE = (120, 120, 120)


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


def _alpha_panel(frame, x1, y1, x2, y2, color=CLR_HUD_BG, alpha=0.72):
    """Semi-transparent filled rectangle for HUD panels."""
    sub = frame[y1:y2, x1:x2]
    if sub.size == 0:
        return
    overlay = np.full(sub.shape, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0, sub)


def _corner_brackets(frame, x1, y1, x2, y2, color, length=14, thickness=2):
    """Draw L-shaped reticle brackets at the four corners of a box."""
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness, cv2.LINE_AA)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness, cv2.LINE_AA)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness, cv2.LINE_AA)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness, cv2.LINE_AA)


def _status(targets: list[Target]) -> tuple[str, tuple[int, int, int]]:
    """Determine the current system status from targets."""
    if not targets:
        return "SCANNING", CLR_STATUS_WARN
    if any(t.predicted is not None for t in targets):
        return "TRACKING", CLR_STATUS_OK
    return "DETECTING", CLR_STATUS_WARN


def _put(frame, text, org, scale=0.5, color=CLR_HUD_TEXT, thickness=1):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, targets: list[Target],
                 predictor: KinematicPredictor,
                 fps: float, frame_num: int, paused: bool = False):
    """Presentation-format HUD: top status bar, per-target reticle overlays,
    bottom telemetry panel for the locked target."""
    h, w = frame.shape[:2]

    # ── Per-target overlays (drawn first so HUD sits on top) ──────────
    locked: Target | None = None
    if targets:
        # Lock onto the highest-confidence target with a prediction,
        # falling back to highest confidence overall.
        with_pred = [t for t in targets if t.predicted is not None]
        pool = with_pred if with_pred else targets
        locked = max(pool, key=lambda t: t.confidence)

    for t in targets:
        x1, y1, x2, y2 = t.bbox
        cx, cy = t.center
        is_locked = (t is locked)

        box_color = CLR_PRED if is_locked else CLR_BBOX

        # Reticle-style corner brackets instead of full rectangle
        _corner_brackets(frame, x1, y1, x2, y2, box_color,
                         length=18 if is_locked else 14,
                         thickness=2)

        # Label panel above the box
        label = f"TGT-{t.track_id:03d}  {t.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly2 = max(y1 - 4, th + 6)
        ly1 = ly2 - th - 6
        _alpha_panel(frame, x1, ly1, x1 + tw + 10, ly2)
        _put(frame, label, (x1 + 5, ly2 - 4), 0.5, box_color, 1)

        # Trail
        trail = predictor.get_trail(t.track_id)
        if len(trail) >= 2:
            for i in range(1, len(trail)):
                alpha = (i + 1) / len(trail)
                color = tuple(int(c * alpha) for c in CLR_TRAIL)
                cv2.line(frame, trail[i - 1], trail[i], color,
                         max(1, int(2 * alpha)), cv2.LINE_AA)

        # Prediction marker + intercept reticle
        if t.predicted:
            px, py = t.predicted
            cv2.line(frame, (cx, cy), (px, py), CLR_PRED, 1, cv2.LINE_AA)
            cv2.line(frame, (px - 14, py), (px + 14, py), CLR_PRED, 2, cv2.LINE_AA)
            cv2.line(frame, (px, py - 14), (px, py + 14), CLR_PRED, 2, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 8, CLR_PRED, 1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 2, CLR_PRED, -1, cv2.LINE_AA)

    # ── TOP HUD BAR ───────────────────────────────────────────────────
    bar_h = 54
    _alpha_panel(frame, 0, 0, w, bar_h, CLR_HUD_BG, 0.78)
    cv2.line(frame, (0, bar_h), (w, bar_h), CLR_HUD_BORDER, 1, cv2.LINE_AA)

    # Title
    _put(frame, "V.A.T.S.", (12, 24), 0.75, CLR_HUD_BRIGHT, 2)
    _put(frame, "VISUAL AERIAL TARGETING SYSTEM", (12, 44), 0.4, CLR_HUD_DIM, 1)

    # Status pill (centered)
    status_text, status_color = _status(targets)
    if paused:
        status_text, status_color = "PAUSED", CLR_STATUS_IDLE
    (sw, sh), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    pill_w = sw + 28
    pill_x = (w - pill_w) // 2
    pill_y = 12
    cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + 30),
                  CLR_HUD_BG, -1)
    cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + 30),
                  status_color, 2, cv2.LINE_AA)
    _put(frame, status_text, (pill_x + 14, pill_y + 22), 0.6, status_color, 2)

    # Right-side stats
    stats_lines = [
        f"FPS  {fps:5.1f}",
        f"TGT  {len(targets):>3}",
        f"FRM  {frame_num:>6}",
    ]
    for i, line in enumerate(stats_lines):
        (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        _put(frame, line, (w - tw - 12, 18 + i * 14), 0.45, CLR_HUD_TEXT, 1)

    # ── BOTTOM TELEMETRY PANEL ────────────────────────────────────────
    panel_h = 76
    py1 = h - panel_h
    _alpha_panel(frame, 0, py1, w, h, CLR_HUD_BG, 0.78)
    cv2.line(frame, (0, py1), (w, py1), CLR_HUD_BORDER, 1, cv2.LINE_AA)

    if locked is not None:
        _put(frame, f"LOCK  TGT-{locked.track_id:03d}",
             (12, py1 + 20), 0.6, CLR_HUD_BRIGHT, 2)

        cx, cy = locked.center
        if locked.predicted:
            px, py = locked.predicted
            spd = f"{locked.speed_px:5.1f} px/f"
            hdg = f"{math.degrees(locked.heading_rad):+6.1f}°"
            cone = f"{math.degrees(locked.cone_spread_rad):4.1f}°"
            pred_str = f"({px:>4},{py:>4})"
        else:
            spd = "  --- px/f"
            hdg = "  ----°"
            cone = "----°"
            pred_str = "(----,----)"

        col1_x = 12
        col2_x = 200
        col3_x = 380
        col4_x = 520

        # Row 1
        _put(frame, "CONF",  (col1_x, py1 + 42), 0.4, CLR_HUD_DIM, 1)
        _put(frame, f"{locked.confidence:.2f}",
             (col1_x + 44, py1 + 42), 0.5, CLR_HUD_TEXT, 1)

        _put(frame, "POS",   (col2_x, py1 + 42), 0.4, CLR_HUD_DIM, 1)
        _put(frame, f"({cx:>4},{cy:>4})",
             (col2_x + 36, py1 + 42), 0.5, CLR_HUD_TEXT, 1)

        _put(frame, "SPD",   (col3_x, py1 + 42), 0.4, CLR_HUD_DIM, 1)
        _put(frame, spd, (col3_x + 36, py1 + 42), 0.5, CLR_HUD_TEXT, 1)

        # Row 2
        _put(frame, "PRED",  (col1_x, py1 + 64), 0.4, CLR_HUD_DIM, 1)
        _put(frame, pred_str,
             (col1_x + 44, py1 + 64), 0.5, CLR_HUD_TEXT, 1)

        _put(frame, "HDG",   (col2_x, py1 + 64), 0.4, CLR_HUD_DIM, 1)
        _put(frame, hdg, (col2_x + 36, py1 + 64), 0.5, CLR_HUD_TEXT, 1)

        _put(frame, "CONE",  (col3_x, py1 + 64), 0.4, CLR_HUD_DIM, 1)
        _put(frame, cone, (col3_x + 44, py1 + 64), 0.5, CLR_HUD_TEXT, 1)
    else:
        _put(frame, "NO LOCK", (12, py1 + 24), 0.6, CLR_HUD_DIM, 2)
        _put(frame, "Awaiting target acquisition...",
             (12, py1 + 50), 0.45, CLR_HUD_DIM, 1)

    # Hotkey legend (bottom-right)
    keys = "[Q] quit   [SPACE] pause   [H] hud   [S] snap"
    (kw, _), _ = cv2.getTextSize(keys, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    _put(frame, keys, (w - kw - 12, h - 8), 0.4, CLR_HUD_DIM, 1)


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
    paused = False
    hud_visible = True
    snap_dir = Path("screenshots")
    last_frame: np.ndarray | None = None
    last_targets: list[Target] = []

    try:
        while True:
            t0 = time.perf_counter()

            if not paused:
                frame = ingest.read()
                if frame is None:
                    print("[VATS] End of stream")
                    break
                frame_num += 1

                # ── Pipeline stages ─────────────────────────────────────
                targets = detector.process(frame)        # Stages 2+3
                predictor.update(targets)                 # Stage 4

                # ── Telemetry output (Stage 5) ──────────────────────────
                for t in targets:
                    print(f"   {format_telemetry_line(t)}")

                # ── FPS measurement ─────────────────────────────────────
                elapsed = time.perf_counter() - t0
                fps_smooth = 0.9 * fps_smooth + 0.1 / max(elapsed, 1e-6)

                last_frame = frame
                last_targets = targets
            else:
                # Re-render the last frame so the HUD/pause state stays live
                if last_frame is None:
                    continue
                frame = last_frame.copy()
                targets = last_targets

            # ── Visualization ───────────────────────────────────────────
            if (show or writer) and hud_visible:
                draw_overlay(frame, targets, predictor, fps_smooth,
                             frame_num, paused=paused)

            if writer and not paused:
                writer.write(frame)

            if show:
                cv2.imshow("V.A.T.S.", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("\n[VATS] Operator terminated")
                    break
                elif key == ord(" "):
                    paused = not paused
                    print(f"[VATS] {'PAUSED' if paused else 'RESUMED'}")
                elif key == ord("h"):
                    hud_visible = not hud_visible
                elif key == ord("s"):
                    snap_dir.mkdir(exist_ok=True)
                    snap_path = snap_dir / f"vats_{int(time.time())}_{frame_num:06d}.png"
                    cv2.imwrite(str(snap_path), frame)
                    print(f"[VATS] Screenshot saved: {snap_path}")

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
