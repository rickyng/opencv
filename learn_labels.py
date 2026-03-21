"""Learning mode — batch frame scanner.

Scans every frame of the video, detects dark anchor bands, extracts all label
regions between them, and produces:
  - learn_labels_results.csv   : all frames sorted by SSIM descending
  - learn_labels_contact.png   : contact sheet of the top-N matching crops
  - learn_labels_summary.png   : verification sheet — frame 506 annotated +
                                  all unique label crops extracted by anchor detection

Usage:
    python learn_labels.py
"""

import csv
import math
from collections import defaultdict
import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False
    print("[WARN] scikit-image not found — using NCC fallback for SSIM.")


# ── Config (keep in sync with label_inspector.py) ────────────────────────────
VIDEO_PATH           = "demo.mp4"
REFERENCE_PATH       = "golden_reference.png"
camera_pixels_per_cm = 59
label_width_cm       = 4.0
label_height_cm      = 4.0
roi_top_offset_cm    = 4.5
SSIM_MATCH_THRESHOLD = 0.75   # frames at or above this are candidate label hits
CONTACT_SHEET_TOPN   = 20    # how many top crops to show in the contact sheet
MATCH_THRESHOLD      = 0.25  # minimum template-match score to accept a detection

# ── Anchor detection config ───────────────────────────────────────────────────
ANCHOR_BRIGHTNESS_THR = 152   # row mean below this → part of a dark anchor band
ANCHOR_MIN_HEIGHT_PX  = 3     # minimum rows to count as a real anchor band
LABEL_MIN_HEIGHT_PX   = 20    # minimum gap height to count as a label region
VERIFY_FRAME_NO       = 506   # frame used for the verification summary PNG
# ─────────────────────────────────────────────────────────────────────────────

LABEL_W_PX = int(label_width_cm  * camera_pixels_per_cm)
LABEL_H_PX = int(label_height_cm * camera_pixels_per_cm)
ROI_TOP_PX  = int(roi_top_offset_cm * camera_pixels_per_cm)

OUT_CSV     = "learn_labels_results.csv"
OUT_CONTACT = "learn_labels_contact-{}.png"  # {} replaced with page number
OUT_SUMMARY = "learn_labels_summary.png"


# =============================================================================
#  HELPERS
# =============================================================================

def crop_roi(frame: np.ndarray) -> "np.ndarray | None":
    fh, fw = frame.shape[:2]
    x0 = (fw - LABEL_W_PX) // 2
    y0 = ROI_TOP_PX
    x1 = x0 + LABEL_W_PX
    y1 = y0 + LABEL_H_PX
    if x0 < 0 or y0 < 0 or x1 > fw or y1 > fh:
        return None
    return frame[y0:y1, x0:x1]


def compute_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Return SSIM score in [0, 1] between two BGR images of the same size."""
    a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    if SKIMAGE_OK:
        score, _ = skimage_ssim(a, b, full=True)
    else:
        # Normalised cross-correlation fallback
        a_f = a.astype(np.float32) - a.mean()
        b_f = b.astype(np.float32) - b.mean()
        denom = np.sqrt((a_f ** 2).sum() * (b_f ** 2).sum())
        score = float((a_f * b_f).sum() / denom) if denom > 0 else 0.0
    return float(score)


def build_contact_sheet(
    results: list,       # (frame_no, ts, position, match_score, ssim_score, crop)
    max_frames: int = 20,
    thumb_w: int = 240,
    thumb_h: int = 240,
    label_h: int = 30,
) -> np.ndarray:
    """Two-column paired layout: each row is one frame, left=above, right=below."""
    by_frame = defaultdict(dict)
    for fn, ts, pos, ms, ss, crop in results:
        by_frame[fn][pos] = (ts, ms, ss, crop)
    frames = sorted(by_frame)[:max_frames]

    n_rows = len(frames)
    cell_w = thumb_w
    cell_h = thumb_h + label_h
    sheet_w = cell_w * 2
    sheet_h = cell_h * n_rows
    sheet = np.zeros((max(sheet_h, 1), max(sheet_w, 1), 3), dtype=np.uint8)

    placeholder = np.full((thumb_h, thumb_w, 3), 40, dtype=np.uint8)

    for row_idx, fn in enumerate(frames):
        y0 = row_idx * cell_h
        for col_idx, pos in enumerate(("above", "below")):
            x0 = col_idx * cell_w
            if pos in by_frame[fn]:
                ts, ms, ss, crop = by_frame[fn][pos]
                thumb = cv2.resize(crop, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                sheet[y0:y0 + thumb_h, x0:x0 + thumb_w] = thumb
                label_text = f"#{fn} {pos}  match={ms:.3f}  ssim={ss:.3f}"
            else:
                sheet[y0:y0 + thumb_h, x0:x0 + thumb_w] = placeholder
                label_text = f"#{fn} {pos}  --"
            bar = sheet[y0 + thumb_h:y0 + cell_h, x0:x0 + thumb_w]
            bar[:] = (30, 30, 30)
            cv2.putText(
                sheet,
                label_text,
                (x0 + 4, y0 + thumb_h + label_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1,
            )

    return sheet


def find_labels_by_template(
    frame: np.ndarray,
    ref_gray: np.ndarray,
    ref_w: int,
    ref_h: int,
) -> "list[tuple[int, int, float]]":
    """Locate up to 2 label positions in *frame* via template matching + NMS.

    Returns a list of (x, y, match_score) for each detected label, sorted by
    match_score descending.  Only peaks above MATCH_THRESHOLD are returned.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray, ref_gray, cv2.TM_CCOEFF_NORMED)

    detections = []
    result_work = result.copy()
    for _ in range(2):
        _, max_val, _, max_loc = cv2.minMaxLoc(result_work)
        if max_val < MATCH_THRESHOLD:
            break
        x, y = max_loc
        detections.append((x, y, float(max_val)))
        # Suppress a vertical window of ref_h//2 around the peak
        # (smaller window avoids eating a nearby second label)
        suppress_y0 = max(0, y - ref_h // 2)
        suppress_y1 = min(result_work.shape[0], y + ref_h // 2)
        result_work[suppress_y0:suppress_y1, :] = -1.0

    return detections


def detect_anchors_and_labels(frame: np.ndarray) -> tuple:
    """Detect dark anchor bands across the full frame width and return the
    label regions (gaps) between consecutive anchors.

    Returns:
        anchors : list of (y_start, y_end) for each anchor band
        labels  : list of (y_start, y_end) for each label region between anchors
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    row_mean = gray.mean(axis=1)  # shape: (H,)

    dark = row_mean < ANCHOR_BRIGHTNESS_THR  # boolean mask

    # Group consecutive dark rows into bands
    anchors = []
    in_band = False
    band_start = 0
    for y, is_dark in enumerate(dark):
        if is_dark and not in_band:
            band_start = y
            in_band = True
        elif not is_dark and in_band:
            if y - band_start >= ANCHOR_MIN_HEIGHT_PX:
                anchors.append((band_start, y - 1))
            in_band = False
    if in_band and len(dark) - band_start >= ANCHOR_MIN_HEIGHT_PX:
        anchors.append((band_start, len(dark) - 1))

    # Gaps between consecutive anchors = label regions
    labels = []
    for i in range(len(anchors) - 1):
        gap_start = anchors[i][1] + 1
        gap_end   = anchors[i + 1][0] - 1
        if gap_end - gap_start + 1 >= LABEL_MIN_HEIGHT_PX:
            labels.append((gap_start, gap_end))

    return anchors, labels


def build_summary_png(frame: np.ndarray, frame_no: int, anchors: list,
                      labels: list) -> np.ndarray:
    """Build a verification PNG:
      Left  — annotated frame (anchor bands = black rect, labels = red rect)
      Right — extracted label crops tiled vertically
    """
    h, w = frame.shape[:2]
    ann = frame.copy()

    # Draw anchor bands in black outline
    for (ay0, ay1) in anchors:
        cv2.rectangle(ann, (0, ay0), (w - 1, ay1), (0, 0, 0), 2)
        cv2.putText(ann, "anchor", (4, ay0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Draw label regions in red outline
    label_crops = []
    for idx, (ly0, ly1) in enumerate(labels):
        cv2.rectangle(ann, (0, ly0), (w - 1, ly1), (0, 0, 255), 2)
        cv2.putText(ann, f"label {idx+1}", (4, ly0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        label_crops.append(frame[ly0:ly1 + 1, :].copy())

    cv2.putText(ann, f"frame {frame_no}", (4, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Right panel: stack label crops with separators
    PANEL_W = 320
    SEP = 4  # separator height in pixels
    rows = []
    for idx, crop in enumerate(label_crops):
        thumb = cv2.resize(crop, (PANEL_W, max(1, int(crop.shape[0] * PANEL_W / crop.shape[1]))),
                           interpolation=cv2.INTER_AREA)
        # caption bar
        bar = np.zeros((20, PANEL_W, 3), dtype=np.uint8)
        bar[:] = (40, 40, 40)
        cv2.putText(bar, f"Label {idx+1}  h={crop.shape[0]}px", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        rows.append(bar)
        rows.append(thumb)
        rows.append(np.zeros((SEP, PANEL_W, 3), dtype=np.uint8))

    if rows:
        right = np.vstack(rows)
    else:
        right = np.zeros((h, PANEL_W, 3), dtype=np.uint8)
        cv2.putText(right, "No labels found", (4, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    # Pad shorter panel to match heights
    target_h = max(ann.shape[0], right.shape[0])
    if ann.shape[0] < target_h:
        pad = np.zeros((target_h - ann.shape[0], ann.shape[1], 3), dtype=np.uint8)
        ann = np.vstack([ann, pad])
    if right.shape[0] < target_h:
        pad = np.zeros((target_h - right.shape[0], PANEL_W, 3), dtype=np.uint8)
        right = np.vstack([right, pad])

    divider = np.full((target_h, 3, 3), 80, dtype=np.uint8)
    summary = np.hstack([ann, divider, right])

    # Header bar
    header_h = 30
    header = np.zeros((header_h, summary.shape[1], 3), dtype=np.uint8)
    header[:] = (20, 20, 60)
    cv2.putText(header,
                f"learn_labels.py — anchor detection verification  "
                f"(frame {frame_no})  "
                f"{len(anchors)} anchors  {len(labels)} labels found",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 255), 1)
    return np.vstack([header, summary])


# =============================================================================
#  MAIN
# =============================================================================

def main() -> None:
    # --- Load reference ------------------------------------------------------
    ref_img = cv2.imread(REFERENCE_PATH)
    if ref_img is None:
        print(f"[ERROR] Cannot read reference: '{REFERENCE_PATH}'")
        return
    ref_resized = cv2.resize(ref_img, (LABEL_W_PX, LABEL_H_PX),
                             interpolation=cv2.INTER_AREA)
    ref_gray = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2GRAY)
    print(f"Reference : {REFERENCE_PATH}  ({LABEL_W_PX}x{LABEL_H_PX} px)")

    # --- Open video ----------------------------------------------------------
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: '{VIDEO_PATH}'")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video     : {VIDEO_PATH}  {w}x{h}  {fps:.1f} fps  {total} frames")
    print(f"Scanning {total} frames (template match + SSIM)...")

    # --- Scan every frame ----------------------------------------------------
    # results: (frame_no, timestamp_s, position, match_score, ssim_score, crop)
    results = []
    verify_frame = None

    for frame_no in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_no == VERIFY_FRAME_NO:
            verify_frame = frame.copy()

        # Locate labels via template matching
        detections = find_labels_by_template(frame, ref_gray, LABEL_W_PX, LABEL_H_PX)
        if not detections:
            if frame_no % 50 == 0:
                print(f"\r  {frame_no}/{total} frames processed...", end="", flush=True)
            continue

        # Classify above / below by Y coordinate
        detections_sorted_y = sorted(detections, key=lambda d: d[1])
        frame_h = frame.shape[0]
        for rank, (x, y, match_score) in enumerate(detections_sorted_y):
            if len(detections_sorted_y) == 1:
                position = "above" if y < frame_h // 2 else "below"
            else:
                position = "above" if rank == 0 else "below"

            crop = frame[y:y + LABEL_H_PX, x:x + LABEL_W_PX]
            if crop.size == 0:
                continue  # truly empty crop (label fully outside frame)
            # Pad partial crops to full size so SSIM comparison is valid
            if crop.shape[0] < LABEL_H_PX or crop.shape[1] < LABEL_W_PX:
                padded = np.zeros((LABEL_H_PX, LABEL_W_PX, 3), dtype=np.uint8)
                padded[:crop.shape[0], :crop.shape[1]] = crop
                crop = padded
            ssim_score = compute_ssim(crop, ref_resized)
            results.append((frame_no, frame_no / fps, position, match_score, ssim_score, crop))

        if frame_no % 50 == 0:
            print(f"\r  {frame_no}/{total} frames processed...", end="", flush=True)

    cap.release()
    print(f"\r  {total}/{total} frames processed.       ")

    if not results:
        print("[ERROR] No labels found — check MATCH_THRESHOLD or reference image.")
        return

    # --- Write CSV (chronological order) ------------------------------------
    results_chron = sorted(results, key=lambda x: (x[0], x[2]))  # frame_no, position
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_no", "timestamp_s", "position", "match_score", "ssim_score"])
        for frame_no, ts, position, match_score, ssim_score, _ in results_chron:
            writer.writerow([frame_no, f"{ts:.3f}", position,
                             f"{match_score:.4f}", f"{ssim_score:.4f}"])
    print(f"CSV saved : {OUT_CSV}  ({len(results)} rows)")

    # --- Contact sheet — one PNG per page, all frames covered ------------------
    all_frames = sorted(set(fn for fn, *_ in results))
    n_pages = math.ceil(len(all_frames) / CONTACT_SHEET_TOPN)
    for page in range(n_pages):
        page_frames = set(all_frames[page * CONTACT_SHEET_TOPN:(page + 1) * CONTACT_SHEET_TOPN])
        page_entries = [e for e in results if e[0] in page_frames]
        sheet = build_contact_sheet(page_entries, max_frames=CONTACT_SHEET_TOPN)
        out_path = OUT_CONTACT.format(page + 1)
        cv2.imwrite(out_path, sheet)
        print(f"Contact   : {out_path}  ({len(page_frames)} frames)")

    # --- Summary of candidate frames above threshold -------------------------
    candidates = [(fn, ts, pos, ms, ssim) for fn, ts, pos, ms, ssim, _ in results
                  if ssim >= SSIM_MATCH_THRESHOLD]
    candidates.sort(key=lambda x: x[0])   # sort by frame number

    print(f"\n── Summary (SSIM >= {SSIM_MATCH_THRESHOLD}) ──")
    print(f"  Detections total : {len(results)}")
    print(f"  Above threshold  : {len(candidates)}")

    if len(candidates) >= 2:
        candidate_frames = sorted(set(fn for fn, *_ in candidates))
        intervals_frames = [candidate_frames[i+1] - candidate_frames[i]
                            for i in range(len(candidate_frames) - 1)]
        intervals_s      = [iv / fps for iv in intervals_frames]
        median_f = float(np.median(intervals_frames))
        median_s = float(np.median(intervals_s))
        labels_per_min = 60.0 / median_s if median_s > 0 else 0
        print(f"  Median interval  : {median_f:.0f} frames  ({median_s:.2f} s)")
        print(f"  Est. label rate  : {labels_per_min:.1f} labels/min")
    else:
        print("  Not enough candidates to estimate label interval.")
        print(f"  Try lowering SSIM_MATCH_THRESHOLD (currently {SSIM_MATCH_THRESHOLD}).")

    top5 = sorted(results, key=lambda x: x[4], reverse=True)[:5]
    print(f"\nTop 5 detections by SSIM:")
    for frame_no, ts, position, match_score, ssim_score, _ in top5:
        print(f"  frame {frame_no:5d}  t={ts:.2f}s  {position:<5}  match={match_score:.3f}  SSIM={ssim_score:.4f}")

    # --- Verification summary PNG --------------------------------------------
    if verify_frame is not None:
        anchors, label_regions = detect_anchors_and_labels(verify_frame)
        print(f"\n── Anchor detection (frame {VERIFY_FRAME_NO}) ──")
        print(f"  Anchors found : {len(anchors)}")
        for i, (a0, a1) in enumerate(anchors):
            print(f"    anchor {i+1}: y={a0}-{a1}  ({a1-a0+1} px)")
        print(f"  Labels found  : {len(label_regions)}")
        for i, (l0, l1) in enumerate(label_regions):
            print(f"    label  {i+1}: y={l0}-{l1}  ({l1-l0+1} px)")

        summary = build_summary_png(verify_frame, VERIFY_FRAME_NO,
                                    anchors, label_regions)
        cv2.imwrite(OUT_SUMMARY, summary)
        print(f"Summary   : {OUT_SUMMARY}  ({summary.shape[1]}x{summary.shape[0]} px)")
    else:
        print(f"[WARN] Frame {VERIFY_FRAME_NO} not found in video — summary PNG skipped.")


if __name__ == "__main__":
    main()
