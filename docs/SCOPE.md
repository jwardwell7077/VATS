# V.A.T.S. — Prototype V1 Scope

## One-Liner

A local-only video demo that detects a drone, tracks it frame-to-frame,
predicts where it's going with a probability cone, and fires a simulated
trigger when conditions align.

---

## What "Done" Looks Like

A single Python script processes a drone video clip. The output is the
same video played back with these overlays drawn in real time:

1. A bounding box around the drone with a label and confidence score.
2. A trajectory trail showing where the drone has been.
3. A prediction cone projecting where it could be next.
4. A "TRIGGER" event flashes on screen when confidence is high AND the
   predicted position enters an alignment zone.
5. A telemetry log prints to the terminal in real time.

No cloud. No network. No hardware. Just a video in, annotated video out.

---

## In Scope (V1)

| # | Feature                  | Detail                                                    |
|---|--------------------------|-----------------------------------------------------------|
| 1 | Sample Video             | Source or generate a ~15s clip of a drone in flight.       |
| 2 | Detection                | YOLOv8 Nano. Bounding box + class label + confidence %.   |
| 3 | Tracking                 | ByteTrack. Persistent target ID across frames.            |
| 4 | Trajectory Trail         | Draw the last N positions as a fading path behind target. |
| 5 | Prediction Cone          | Project future position as a cone/fan of probable area.   |
| 6 | Simulated Trigger        | Visual + terminal event when confidence > threshold AND   |
|   |                          | predicted position is within the "kill zone."             |
| 7 | Telemetry Log            | Structured stdout: ID, confidence, position, prediction.  |
| 8 | Air-Gapped               | Zero network calls. All models bundled locally.           |
| 9 | Annotated Output         | Option to save the overlaid video as a new .mp4 file.     |

## Out of Scope (V1)

- Real-time webcam / RTSP ingestion (future — architecture supports it).
- Actual hardware trigger or microcontroller handoff.
- Multi-camera / stereoscopic tracking.
- Model training or fine-tuning.
- GUI controls beyond play/quit.

---

## Prediction Cone — How It Works

```
                        ╱  · · ·  ╲
                      ╱  · · · · ·  ╲
          ●━━━━━━━━━◉╱ · · · · · · · ╲
        trail      now   probability    far edge
                         cone           (lookahead)
```

- Store last N center-points in a rolling buffer.
- Compute average velocity vector (dx, dy per frame).
- The cone's center axis = velocity vector projected forward.
- The cone's spread = uncertainty, which widens with distance.
  Based on the variance in the velocity buffer.
- Draw as a filled semi-transparent fan on the frame.

---

## Simulated Trigger Logic

```
IF  detection_confidence  >  TRIGGER_CONFIDENCE   (e.g. 0.80)
AND predicted_center      IN alignment_zone        (center region of frame)
THEN
    flash "LOCKED" overlay on frame
    print TRIGGER event to terminal
```

The alignment zone is a rectangular region in the center of the frame
representing the turret's effective firing arc.

---

## Tech Stack

| Component     | Library                          |
|---------------|----------------------------------|
| Language      | Python 3.10+                     |
| Detection     | ultralytics (YOLOv8 Nano)        |
| Tracking      | ByteTrack (via ultralytics)      |
| Video / Draw  | opencv-python                    |
| Math          | numpy                            |

All pip-installable. No system-level dependencies beyond Python.

---

## Deliverables

```
VATS/
├── vats.py              # The pipeline — single entry point
├── requirements.txt     # pip dependencies
├── samples/             # Test video(s)
│   └── drone_clip.mp4
└── README.md            # Usage instructions
```

---

## Open Questions

- [ ] Video source: Do you have a clip, or should I generate a
      synthetic drone animation with OpenCV for a guaranteed air-gapped
      demo? (No download needed.)
- [ ] Alignment zone: Fixed center rectangle, or configurable?
- [ ] Trigger cooldown: Should it fire once per pass, or every frame
      the conditions hold?
