# Label Inspector

Real-time label inspection for flexo/digital label presses via USB camera.
Built with OpenCV, NumPy, and scikit-image — no deep learning required.

Each camera frame is scanned for **two labels** (above and below) using template
matching. Both are compared against the golden reference and independently
classified as PASS or FAIL.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  USB Camera (or video file)                                     │
│      │                                                          │
│      ▼                                                          │
│  CameraThread  (daemon)                                         │
│      │  bounded queue  maxsize=2  (drops stale frames)          │
│      ▼                                                          │
│  crop_labels()  — template matching, up to 2 detections/frame  │
│      │  returns {"above": crop, "below": crop}                  │
│      ▼                                                          │
│  compare_labels()  — per crop                                   │
│      ├─ Metric 1: abs-diff defect pixel ratio                   │
│      ├─ Metric 2: SSIM score  (scikit-image / NCC fallback)     │
│      └─ Metric 3: template-match score  (warning only)          │
│      │                                                          │
│      ├── PASS ──▶  session stats update                         │
│      └── FAIL ──▶  save_defect() + log_defect_csv()             │
│                                                                  │
│  build_display()  →  cv2.imshow()  (3 × 2 panel grid)          │
│      ┌─────────────┬─────────────┬─────────────┐               │
│      │ A: Full     │ B: Above    │ C: Below    │               │
│      │ frame + ROI │ crop        │ crop        │               │
│      ├─────────────┼─────────────┼─────────────┤               │
│      │ D: Stats +  │ E: Above    │ F: Below    │               │
│      │ reference   │ diff heatmap│ diff heatmap│               │
│      └─────────────┴─────────────┴─────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

### Display panels

| Panel | Contents |
|---|---|
| A | Full camera frame with ROI rectangles for above and below labels |
| B | Current above-label crop (resized to panel dimensions) |
| C | Current below-label crop (or placeholder if not detected) |
| D | Golden reference thumbnail + session statistics (frames, pass rates) |
| E | Above-label abs-diff heatmap (green = match, red = defect) |
| F | Below-label abs-diff heatmap |

### Module reference

| Symbol | Responsibility |
|---|---|
| `CameraThread` | Daemon thread; drops stale frames to prevent lag build-up |
| `crop_labels()` | Template matching (TM_CCOEFF_NORMED) → up to 2 label detections per frame, classified as above/below by Y position |
| `compare_labels()` | Abs-diff ratio + SSIM + template-match → PASS/FAIL per label |
| `build_display()` | Assembles 3×2 panel composite every frame |
| `load_reference()` / `save_reference()` | Load/overwrite the golden master label image |
| `save_defect()` | Saves PNG crop + diff heatmap; filename encodes timestamp, frame number, and position |
| `log_defect_csv()` | Appends one row per defect to `defects_log.csv` |
| `main()` | FPS counter (EMA), key handlers, shutdown sequence |

### Threading model

```
Thread 1 — CameraThread (daemon)
    cap.read() → frame_queue.put()

Thread 2 (main)
    frame_queue.get() → crop_labels() → compare_labels() → build_display()
```

The two threads share only the bounded `queue.Queue(maxsize=2)`.
No locks are needed; the queue itself is thread-safe.

---

## Scripts

### `label_inspector.py` — real-time inspection

Main application. Reads frames from a USB camera (or video file for testing),
detects both above and below labels per frame via template matching, compares
each against the golden reference, and displays a live 3×2 panel view.

### `learn_labels.py` — offline batch scanner

Scans every frame of a video file to:
- Locate all label detections (above + below) via template matching
- Compute SSIM score vs. the golden reference for each crop
- Write results to `learn_labels_results.csv`
- Build a paginated contact sheet (`learn_labels_contact-1.png`, `-2.png`, …)
  with paired above/below columns per row
- Generate a verification PNG (`learn_labels_summary.png`) showing anchor
  detection on a chosen frame

Use this script to calibrate thresholds before going live.

---

## Quick-start

### 1. Install dependencies

```bash
pip install opencv-python numpy scikit-image
```

### 2. Connect the USB camera and verify the index

```bash
# Linux
ls /dev/video*

# Windows — try CAMERA_INDEX = 0, 1, 2 … in the config section
```

### 3. Set user-configurable variables at the top of `label_inspector.py`

```python
CAMERA_INDEX         = 0       # camera index or path to a video file
label_width_cm       = 4.0     # physical label width (cross-web)
label_height_cm      = 4.0     # physical label height / pitch (web direction)
camera_pixels_per_cm = 150     # ← CALIBRATE THIS FIRST (see below)
roi_top_offset_cm    = 1.0     # vertical ROI offset from top of frame
```

### 4. First run

```bash
python label_inspector.py
```

- A blank (black) reference is used until you save one.
- Run the press over a known-good label.
- Press **`s`** to capture and save it as `golden_reference.png`.

### 5. Ongoing operation

| Key | Action |
|---|---|
| `s` | Save current crop as new golden reference |
| `q` | Quit the application |

### 6. Output files

| Path | Contents |
|---|---|
| `inspection.log` | Timestamped log of every label result |
| `defects_log.csv` | CSV: timestamp, frame, position, SSIM, defect %, tmatch, filename |
| `defects/<name>_crop.png` | Raw crop of each failed label |
| `defects/<name>_diff.png` | Difference heatmap for each failed label |
| `golden_reference.png` | Current golden reference image |

Defect filenames include timestamp, frame number, and position, e.g.:
`defect_20260321_120000_000_frame000042_above_label000042_crop.png`

---

## Calibration checklist

Complete these steps **before** going live. Repeat whenever the lens zoom,
mount height, or camera model changes.

### Step 1 — Measure pixels per cm

1. Tape a flat mm ruler across the camera field-of-view, parallel to the
   label edge (cross-web direction).
2. Start the program and press **`s`** to capture a frame, or grab one with
   any other tool.
3. Open the image in a viewer that shows pixel coordinates (e.g. GIMP,
   ImageJ, or Paint).
4. Click on the ruler at two marks separated by exactly **10 cm**.
   Record the x-coordinates `x1` and `x2`.
5. Calculate:
   ```
   camera_pixels_per_cm = abs(x2 - x1) / 10
   ```
6. Set this value in `label_inspector.py`.

> **Example:** marks at x=142 and x=1642 → `(1642 − 142) / 10 = 150 px/cm`

### Step 2 — Verify label pixel dimensions

After setting `camera_pixels_per_cm`, confirm the derived sizes look right:

```
LABEL_W_PX = label_width_cm  × camera_pixels_per_cm
LABEL_H_PX = label_height_cm × camera_pixels_per_cm
```

The startup log line confirms these:
```
Label size: 236×236 px  (4.0×4.0 cm @ 59 px/cm)
```

### Step 3 — Align the ROI

1. Run the program and watch **panel A** (full frame).
2. Two coloured rectangles show the above and below inspection ROIs.
3. Adjust `roi_top_offset_cm` until the boxes sit cleanly over the labels.

### Step 4 — Save a golden reference

1. Run the press at normal speed with a **verified defect-free** label.
2. Watch the live crop panel (panel B or C).
3. When a well-aligned label is visible, press **`s`**.
4. Confirm the log shows:
   ```
   New golden reference saved to 'golden_reference.png'.
   ```

### Step 5 — Tune thresholds

Run 20–30 known-good labels and note the SSIM scores in `inspection.log`.
Then adjust:

| Variable | Default | Guidance |
|---|---|---|
| `SSIM_FAIL_THRESHOLD` | `0.92` | Lower if too many false positives; raise for tighter QC |
| `DEFECT_PIXEL_RATIO` | `0.05` | 5 % of pixels; lower for finer defect sensitivity |
| `DIFF_BINARIZE_THR` | `30` | Pixel-value delta to call a pixel defective (0–255) |
| `TMATCH_WARN_THRESHOLD` | `0.80` | Warning only; does not affect PASS/FAIL |
| `MATCH_THRESHOLD` | `0.25` | Minimum template-match score to accept a detection |

### Step 6 — Verify at speed

1. Run the press at **production speed**.
2. Monitor FPS in panel D — it should stay above the frame rate needed to
   cover one label pitch without missing a detection.
3. If FPS drops, reduce `CAMERA_WIDTH_PX` / `CAMERA_HEIGHT_PX`.
4. If labels are not detected, lower `MATCH_THRESHOLD` slightly (e.g. 0.20).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Cannot open camera index 0` | Wrong index or driver missing | Try `CAMERA_INDEX = 1` or install camera driver |
| ROI out of bounds warning | Calibration mismatch | Reduce `camera_pixels_per_cm` or `label_*_cm` |
| No labels detected | `MATCH_THRESHOLD` too high or wrong reference | Lower `MATCH_THRESHOLD`; ensure `golden_reference.png` exists |
| Constant FAILs on good labels | Reference not saved yet | Press `s` on a known-good label |
| High false-positive rate | Threshold too tight | Increase `SSIM_FAIL_THRESHOLD` or `DEFECT_PIXEL_RATIO` |
| Only one label detected per frame | Label partially outside frame or low match score | Check `roi_top_offset_cm`; lower `MATCH_THRESHOLD` |