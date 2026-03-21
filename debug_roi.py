"""Debug script: visualise ROI on a mid-video frame.

Usage:
    python debug_roi.py [frame_number]
    python debug_roi.py crop [frame_number]

If frame_number is omitted, the middle frame of the video is used.

Default mode:
    Outputs debug_roi_frameN.png (full frame + green ROI box)
         and debug_roi_cropN.png  (cropped ROI region).

crop mode:
    Extracts the ROI crop and saves it as the golden reference
    for label_inspector.py (golden_label.jpeg).
"""

import sys
import cv2
import numpy as np

# ── Config (keep in sync with label_inspector.py) ────────────────────────────
VIDEO_PATH           = "demo.mp4"
REFERENCE_PATH       = "golden_label.jpeg"  # must match label_inspector.py reference_image_path
camera_pixels_per_cm = 59
label_width_cm       = 4.1
label_height_cm      = 3.7
roi_top_offset_cm    = 4.9
# ─────────────────────────────────────────────────────────────────────────────

LABEL_W_PX = int(label_width_cm  * camera_pixels_per_cm)
LABEL_H_PX = int(label_height_cm * camera_pixels_per_cm)
ROI_TOP_PX  = int(roi_top_offset_cm * camera_pixels_per_cm)


def load_frame(video_path: str, frame_number: int | None = None) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = frame_number if frame_number is not None else total // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {idx} from {video_path}")
    print(f"Read frame {idx}/{total} from {video_path}")
    return frame


def transform_frame(frame: np.ndarray) -> np.ndarray:
    """Apply the same transform as CameraThread: rotate 90° CW + resize to 240×360."""
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame = cv2.resize(frame, (240, 360))
    return frame


def draw_roi(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    fh, fw = frame.shape[:2]
    roi_left   = (fw - LABEL_W_PX) // 2
    roi_top    = ROI_TOP_PX
    roi_right  = roi_left + LABEL_W_PX
    roi_bottom = roi_top  + LABEL_H_PX

    print(f"Frame : {fw}x{fh}")
    print(f"ROI   : left={roi_left}, top={roi_top}, right={roi_right}, bottom={roi_bottom}")

    if roi_left < 0 or roi_top < 0 or roi_right > fw or roi_bottom > fh:
        print("WARNING: ROI out of bounds!")

    debug = frame.copy()
    #cv2.rectangle(debug, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
    return debug, (roi_left, roi_top, roi_right, roi_bottom)


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "crop":
        save_as_reference = True
        frame_number = int(args[1]) if len(args) > 1 else None
    else:
        save_as_reference = False
        frame_number = int(args[0]) if args else None

    frame = load_frame(VIDEO_PATH, frame_number)
    #frame = transform_frame(frame)

    debug, (l, t, r, b) = draw_roi(frame)
    crop = frame[t:b, l:r]

    if save_as_reference:
        cv2.imwrite(REFERENCE_PATH, crop)
        print(f"Saved golden reference -> {REFERENCE_PATH}  ({crop.shape[1]}x{crop.shape[0]} px)")
    else:
        suffix = frame_number if frame_number is not None else "mid"
        out_full = f"debug_roi_frame_{suffix}.png"
        out_crop = f"debug_roi_crop_{suffix}.png"
        cv2.imwrite(out_full, debug)
        cv2.imwrite(out_crop, crop)
        print(f"Saved {out_full} and {out_crop}")


if __name__ == "__main__":
    main()
