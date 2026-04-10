# V.A.T.S. — Visual Aerial Targeting System

Real-time, edge-capable computer vision sensor for drone detection and
kinematic prediction. Runs 100% locally. Zero network calls. Air-gapped
by design.

V.A.T.S. is a **sensor**. It outputs structured telemetry — current
position, predicted future position, speed, and an uncertainty cone. A
downstream system is responsible for fusing that with RF data and rules
of engagement to make any actual engagement decision. V.A.T.S. itself
makes no fire-control decisions.

## Pipeline

```
Webcam → YOLOv8n → ByteTrack → Kinematic Predictor → Telemetry
         (detect)  (track ID)   (velocity + cone)    (stdout + HUD)
```

| # | Stage | Component |
|---|-------|-----------|
| 1 | Ingest    | OpenCV V4L2 capture (MJPEG forced for usbipd) |
| 2 | Detect    | YOLOv8 Nano, fine-tuned on a custom dataset |
| 3 | Track     | ByteTrack — Kalman filter + Hungarian assignment, persistent IDs across frames |
| 4 | Predict   | Rolling position buffer → mean velocity → projected position + angular-uncertainty cone |
| 5 | Telemetry | One structured line per target per frame, plus visual HUD overlay |

## Quickstart

```bash
git clone <this repo> && cd VATS
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the live demo against your webcam (camera index 0)
python3 src/vats.py 0
```

The default model is `models/vats_drone.pt` — a YOLOv8 Nano fine-tuned
on a small custom dataset captured specifically for this demo.

## Telemetry output

```
[TGT-001] CONF 0.87 | POS ( 423, 312) | PRED ( 445, 298) | SPD  12.3 px/f | HDG  -45.2° | CONE  18.5°
```

Each line is a complete sensor reading for one tracked target on one
frame. A downstream system can consume this directly — no need to
re-parse the visual frame.

## Visual overlay

- Green bounding box with `TGT-XXX` ID and confidence
- Cyan trail dots showing recent positions
- Red prediction line + crosshair at projected intercept
- HUD bar with FPS, target count, frame number

## Layout

```
VATS/
├── src/
│   └── vats.py               # the live pipeline (the demo entry point)
├── scripts/                  # dataset + training tools
│   ├── capture.py            # record clips from the webcam
│   ├── extract_frames.py     # pull frames from raw videos
│   ├── auto_label.py         # pseudo-label frames with an existing model
│   ├── review.py             # manually delete bad pseudo-labels
│   ├── prepare_dataset.py    # split into train/val and write data.yaml
│   ├── train.py              # fine-tune YOLOv8n
│   └── download_dataset.py   # (legacy) Roboflow downloader
├── docs/
│   ├── SCOPE.md              # locked V1 scope
│   ├── ARCHITECTURE.md       # pipeline diagrams
│   └── SETUP.md              # WSL2 + usbipd + GPU setup
├── models/
│   ├── vats_drone.pt         # the fine-tuned demo model
│   └── yolov8n.pt            # YOLOv8 Nano base (for retraining)
├── requirements.txt
└── README.md
```

All scripts anchor their relative paths to the repo root, so they work
no matter where you invoke them from.

## Retraining workflow

```bash
# 1. Record a few clips of the drone in your scene
python3 scripts/capture.py

# 2. Pull frames out (every Nth to avoid near-duplicates)
python3 scripts/extract_frames.py

# 3. Pseudo-label using the current model (high conf for clean labels)
python3 scripts/auto_label.py --conf 0.7

# 4. Step through and delete the bad ones
python3 scripts/review.py

# 5. Split into train/val and write data.yaml
python3 scripts/prepare_dataset.py

# 6. Fine-tune YOLOv8n (~20 min on an RTX 4070)
python3 scripts/train.py

# 7. Test
python3 src/vats.py 0
```

## Tech stack

| Component | Library |
|-----------|---------|
| Detection | ultralytics (YOLOv8 Nano) |
| Tracking  | ByteTrack (via ultralytics) |
| Video I/O | opencv-python |
| Math      | numpy |

## Keybinds

- `q` / `ESC` — quit
- `SPACE` — pause / resume the feed (HUD stays live)
- `h` — toggle HUD overlay (clean view of the raw frame)
- `s` — save annotated screenshot to `screenshots/`
