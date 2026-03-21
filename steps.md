# Step-by-Step Guide

This guide walks through:
1. Using `debug_roi.py` to verify ROI alignment and produce a golden reference
2. Running `label_inspector.py` for live inspection

---

## Step 1 — Verify your config values

Before running anything, open `label_inspector.py` and set these values to match
your physical setup:

```python
camera_pixels_per_cm = 59      # measure with a ruler (see README calibration)
label_width_cm       = 4.0     # physical label width (cross-web)
label_height_cm      = 4.0     # physical label height / pitch
roi_top_offset_cm    = 4.5     # distance from top of frame to label
```

Set the same values in `debug_roi.py` — the comment at the top of the file
reminds you to keep them in sync.

---

## Step 2 — Check ROI alignment with `debug_roi.py`

`debug_roi.py` grabs a single frame from the video (or camera feed), draws the
ROI box on it, and saves two PNG files so you can verify alignment without
running the full inspector.

### 2a. Visualise ROI on the mid-frame

```bash
python debug_roi.py
```

Outputs:
- `debug_roi_frame_mid.png` — full frame with a green rectangle showing the ROI
- `debug_roi_crop_mid.png` — the cropped region alone

Open these images and confirm the green box sits squarely over one label.

### 2b. Visualise ROI on a specific frame

```bash
python debug_roi.py 506
```

Outputs `debug_roi_frame_506.png` and `debug_roi_crop_506.png`.

Use this to check alignment at a frame where you know a label is present.

### 2c. Adjust if the box is misaligned

| Problem | Fix |
|---|---|
| Box too high (above the label) | Increase `roi_top_offset_cm` |
| Box too low (below the label) | Decrease `roi_top_offset_cm` |
| Box too wide / too narrow | Adjust `label_width_cm` |
| Box too tall / too short | Adjust `label_height_cm` |
| Everything too small / too large | Re-measure `camera_pixels_per_cm` |

Repeat until `debug_roi_crop_mid.png` shows exactly one clean label.

---

## Step 3 — Save a golden reference with `debug_roi.py`

Once the ROI is aligned, extract the crop from a known-good frame and save it
as the golden reference image that `label_inspector.py` will compare against.

```bash
python debug_roi.py crop 506
```

This saves the crop as `golden_reference.png` (the filename `label_inspector.py`
looks for by default).

You can choose any frame number where a clean, defect-free label is visible.
If you omit the frame number, the mid-frame is used:

```bash
python debug_roi.py crop
```

> **Tip:** You can also save the reference at any time during live inspection
> by pressing **`s`** — the current above-label crop is saved as the new
> `golden_reference.png`.

---

## Step 4 — Run `label_inspector.py`

```bash
python label_inspector.py
```

The inspector opens a window with a **3 × 2 panel display** and begins
processing frames immediately.

### Panel layout

```
┌──────────────────┬──────────────────┬──────────────────┐
│  A: Full frame   │  B: Above crop   │  C: Below crop   │
│  + ROI overlays  │                  │                  │
├──────────────────┼──────────────────┼──────────────────┤
│  D: Stats panel  │  E: Above diff   │  F: Below diff   │
│  + reference img │     heatmap      │     heatmap      │
└──────────────────┴──────────────────┴──────────────────┘
```

| Panel | Description |
|---|---|
| **A** | Full camera frame. Two rectangles overlay the detected above and below label positions. Green = PASS, red = FAIL. |
| **B** | Current above-label crop resized to the panel. Labelled with PASS/FAIL and SSIM score. |
| **C** | Current below-label crop. Shows a dark placeholder when no below label is detected. |
| **D** | Golden reference thumbnail (top) and live session statistics (bottom): frames processed, frames with both labels, above pass %, below pass %. |
| **E** | Abs-diff heatmap for the above label. Bright areas indicate pixel-level differences from the reference. |
| **F** | Abs-diff heatmap for the below label. |

### Keyboard shortcuts

| Key | Action |
|---|---|
| `s` | Save the current above-label crop as a new `golden_reference.png` |
| `q` | Quit the application |

---

## Step 5 — Review output files

| File / Folder | Description |
|---|---|
| `inspection.log` | Timestamped log of every frame result (PASS/FAIL, scores) |
| `defects_log.csv` | CSV with one row per defect: timestamp, frame, position, SSIM score, defect pixel %, template-match score, filename |
| `defects/` | Folder containing PNG pairs for every failed label |
| `golden_reference.png` | Current golden reference image |

### Defect filename format

Each defect produces two files:

```
defects/defect_<timestamp>_frame<NNNNNN>_<position>_label<NNNNNN>_crop.png
defects/defect_<timestamp>_frame<NNNNNN>_<position>_label<NNNNNN>_diff.png
```

Example:
```
defects/defect_20260321_120000_000_frame000042_above_label000042_crop.png
defects/defect_20260321_120000_000_frame000042_above_label000042_diff.png
```

- `_crop.png` — the raw label crop at the time of failure
- `_diff.png` — the abs-diff heatmap (same view as panel E or F)

---

## Optional — Offline batch scan with `learn_labels.py`

Before deploying the live inspector, use `learn_labels.py` to scan the entire
demo video and check that detections look correct:

```bash
python learn_labels.py
```

Outputs:

| File | Description |
|---|---|
| `learn_labels_results.csv` | All detections: frame, timestamp, position, match score, SSIM score |
| `learn_labels_contact-1.png`, `-2.png`, … | Paginated contact sheets — each row is one frame, left column = above crop, right column = below crop |
| `learn_labels_summary.png` | Verification PNG: annotated frame showing anchor bands and label regions |

Review the contact sheets to confirm both above and below labels are detected
cleanly across the video before going live.
