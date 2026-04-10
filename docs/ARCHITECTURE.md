# V.A.T.S. — System Architecture

## V1 Prototype (Current Build)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        V.A.T.S. PIPELINE — V1                              │
│                                                                             │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐  │
│   │  VIDEO    │    │  STAGE 1     │    │  STAGE 2     │    │  STAGE 3    │  │
│   │  SOURCE   │───►│  PERCEPTION  │───►│  TRACING     │───►│  PREDICTION │  │
│   │          │    │              │    │              │    │  + TRIGGER  │  │
│   │ .mp4     │    │ YOLOv8 Nano  │    │ ByteTrack    │    │ Kinematic   │  │
│   │ webcam   │    │ yolov8n.pt   │    │ Persistent   │    │ Velocity    │  │
│   │ RTSP     │    │              │    │ Target IDs   │    │ Cone        │  │
│   └──────────┘    └──────────────┘    └──────────────┘    └──────┬──────┘  │
│                                                                   │         │
│                          ┌────────────────────────────────────────┘         │
│                          │                                                  │
│                          ▼                                                  │
│               ┌─────────────────────┐     ┌──────────────────────┐         │
│               │  OVERLAY RENDERER   │     │  TELEMETRY STDOUT    │         │
│               │                     │     │                      │         │
│               │  • Bounding boxes   │     │  [TGT-001] CONF 0.92│         │
│               │  • Trail dots       │     │  POS (423,312)       │         │
│               │  • Prediction cone  │     │  PRED (445,298)      │         │
│               │  • Crosshair        │     │  SPD 12.3            │         │
│               │  • Alignment zone   │     │  <<< TRIGGER >>>     │         │
│               │  • TRIGGER flash    │     │                      │         │
│               │  • HUD bar          │     │  (consumable by      │         │
│               └────────┬────────────┘     │   microcontrollers)  │         │
│                        │                  └──────────────────────┘         │
│                        ▼                                                   │
│               ┌─────────────────────┐                                      │
│               │  OUTPUT             │                                      │
│               │  • cv2.imshow()     │                                      │
│               │  • .mp4 file save   │                                      │
│               └─────────────────────┘                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Stage Detail

### Stage 1 — Perception

```
Raw Frame (BGR)
     │
     ▼
┌────────────────────────────┐
│ YOLOv8 Nano Inference      │
│                            │
│  Input:  640x640 tensor    │
│  Output: N detections      │
│          per detection:    │
│            • bbox (xyxy)   │
│            • class_id      │
│            • confidence    │
│                            │
│  Filter: class ∈ {aero,    │
│          bird, person}     │
│          conf > 0.30       │
└────────────┬───────────────┘
             │
             ▼ filtered detections
```

### Stage 2 — Tracing (ByteTrack)

```
Filtered Detections
     │
     ▼
┌────────────────────────────┐
│ ByteTrack Tracker          │
│                            │
│  • Assigns persistent IDs  │
│  • Handles occlusions      │
│  • Survives dropped frames │
│                            │
│  Output per target:        │
│    • track_id (int)        │
│    • bbox (xyxy)           │
│    • center (cx, cy)       │
└────────────┬───────────────┘
             │
             ▼ tracked targets
```

### Stage 3 — Kinematic Prediction + Trigger

```
Tracked Center (cx, cy) per frame
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│ KinematicPredictor                                      │
│                                                         │
│  Rolling Buffer (deque, last 8 positions per target)    │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┐                     │
│  │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ ◄── newest        │
│  └───┴───┴───┴───┴───┴───┴───┴───┘                     │
│                                                         │
│  Velocity:                                              │
│    deltas = diff(buffer)                                │
│    vx, vy = mean(deltas)                                │
│    speed  = √(vx² + vy²)                               │
│                                                         │
│  Direction:                                             │
│    angle = atan2(vy, vx)                                │
│                                                         │
│  Uncertainty (cone spread):                             │
│    angle_std = std(atan2 of each delta)                 │
│    spread = max(8°, angle_std × 1.5)                    │
│    capped at 60°                                        │
│                                                         │
│  Projection:                                            │
│    pred_x = last_x + vx × lookahead                    │
│    pred_y = last_y + vy × lookahead                     │
│                                                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Trigger Logic                                           │
│                                                         │
│  IF  conf ≥ 0.70                                        │
│  AND predicted_center ∈ alignment_zone                  │
│  THEN  >>> TRIGGER <<<                                  │
│                                                         │
│  Alignment Zone:                                        │
│  ┌────────────────────────────────────┐                 │
│  │            30% margin              │                 │
│  │   ┌────────────────────────┐       │                 │
│  │   │                        │ 25%   │                 │
│  │   │     KILL ZONE          │ margin│                 │
│  │   │     (center 40% × 50%) │       │                 │
│  │   │                        │       │                 │
│  │   └────────────────────────┘       │                 │
│  └────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────┘
```

## Data Flow Per Frame

```
                    time ──────────────────────────►

Frame N             Frame N+1           Frame N+2
  │                   │                   │
  ▼                   ▼                   ▼
┌──────┐           ┌──────┐           ┌──────┐
│ YOLO │           │ YOLO │           │ YOLO │
└──┬───┘           └──┬───┘           └──┬───┘
   │                  │                  │
   ▼                  ▼                  ▼
┌──────┐           ┌──────┐           ┌──────┐
│ BYTE │           │ BYTE │           │ BYTE │
│TRACK │           │TRACK │           │TRACK │
└──┬───┘           └──┬───┘           └──┬───┘
   │ id=1             │ id=1             │ id=1
   │ (320,200)        │ (340,195)        │ (362,188)
   ▼                  ▼                  ▼
┌──────┐           ┌──────┐           ┌──────┐
│BUFFER│           │BUFFER│           │BUFFER│
│[0]   │           │[0,1] │           │[0,1,2]
└──┬───┘           └──┬───┘           └──┬───┘
   │                  │                  │
   │ no pred          │ no pred          │ pred!
   │ (need ≥3 pts)    │ (need ≥3 pts)   │ → (405,168)
   │                  │                  │ → cone drawn
   ▼                  ▼                  ▼
┌──────┐           ┌──────┐           ┌──────┐
│ DRAW │           │ DRAW │           │ DRAW │
│ bbox │           │ bbox │           │ bbox │
│ only │           │trail │           │trail │
│      │           │      │           │cone  │
│      │           │      │           │xhair │
│      │           │      │           │TRIGGER│
└──────┘           └──────┘           └──────┘
```

## Dream Architecture (V2 — Future)

```
                    ┌──────────────────────┐
                    │   COMMAND & CONTROL   │
                    │   (Edge Server)       │
                    │                      │
                    │  ┌────────────────┐  │
                    │  │  VATS Engine   │  │
                    │  │  Multi-target  │  │
                    │  │  Fusion        │  │
                    │  └───────┬────────┘  │
                    │          │           │
                    └──────────┼───────────┘
                         ┌─────┼─────┐
                         │     │     │
                    ┌────┴┐ ┌─┴──┐ ┌┴────┐
                    │CAM 1│ │CAM 2│ │CAM 3│   Stereoscopic
                    │ L   │ │ C   │ │ R   │   multi-angle
                    └──┬──┘ └──┬──┘ └──┬──┘
                       │      │      │
                       ▼      ▼      ▼
                    ┌─────────────────────┐
                    │  3D TRIANGULATION   │
                    │  Real-world (x,y,z) │
                    │  coordinates        │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  TURRET CONTROLLER  │
                    │  (Microcontroller)  │
                    │                     │
                    │  Serial / GPIO out  │
                    │  Pan: θ   Tilt: φ   │
                    │  Fire: bool         │
                    └─────────────────────┘
```
