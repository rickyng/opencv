"""
label_inspector.py
══════════════════
Real-time label inspection for flexo/digital label presses via USB camera.

Pipeline:
    CameraThread → frame_queue → SoftwareEncoder (optical-flow trigger)
    → crop_label → correct_skew → compare_labels → visualize / log

Keys during run:
    q  — quit
    s  — save current crop as new golden reference

Dependencies:
    pip install opencv-python numpy scikit-image
"""

# ── Standard library ─────────────────────────────────────────────────────────
import csv
import logging
import os
import queue
import threading
import time
from datetime import datetime

# ── Third-party ──────────────────────────────────────────────────────────────
import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False
    print(
        "[WARN] scikit-image not found — SSIM will use a NCC fallback.\n"
        "       Install with: pip install scikit-image"
    )


# ═════════════════════════════════════════════════════════════════════════════
#  USER-CONFIGURABLE VARIABLES  ← edit these before deployment
# ═════════════════════════════════════════════════════════════════════════════

CAMERA_INDEX         = "demo.mp4"  # USB camera index or video file path for testing
CAMERA_WIDTH_PX      = 1280    # Requested capture resolution (camera may
CAMERA_HEIGHT_PX     = 720     #   clamp to its nearest supported mode)

# Physical label dimensions — measure the finished, printed label
label_width_cm       = 4.0     # Label width  in the cross-web direction (cm)
label_height_cm      = 4.0     # Label height / pitch in the web direction (cm)

# ┌─ CALIBRATION ────────────────────────────────────────────────────────────┐
# │  1. Tape a mm ruler across the camera field-of-view, parallel to the    │
# │     label edge.                                                          │
# │  2. Grab a still frame (press 's' to save one) and open it in any       │
# │     image viewer that shows pixel coordinates.                           │
# │  3. Note the pixel positions of two ruler marks separated by 10 cm.     │
# │  4. camera_pixels_per_cm = pixel_distance / 10.0                        │
# │  5. Re-calibrate whenever lens zoom or mount height changes.             │
# └──────────────────────────────────────────────────────────────────────────┘
camera_pixels_per_cm = 59      # native 1280×720 testing — recalibrate with ruler for real camera

# Vertical offset of the label ROI from the top of the frame (cm).
# Increase if the label appears in the lower half of the frame.
roi_top_offset_cm    = 4.5      # frame is native 1280×720 for landscape video testing

# Golden reference label path.  Press 's' on a good live label to create it.
reference_image_path = "golden_reference.png"

# Video file testing — set CAMERA_INDEX to a file path to use a demo video.
# Example:  CAMERA_INDEX = "demo.mp4"
# VIDEO_LOOP = True  loops the video indefinitely (useful for bench testing).
# VIDEO_LOOP = False stops the program when the video ends.
VIDEO_LOOP = False

# Debug: pause after every label detection and wait for a keypress to continue.
# Press any key (except 'q'/'s') to advance; 'q' still quits.
DEBUG_STEP_MODE = True

# When CAMERA_INDEX is a video file, fire a label trigger every N frames
# instead of using optical flow (which requires real web motion).
# At 30 fps: 90 frames = 3 s per label.  Adjust to match your label pitch.
VIDEO_TRIGGER_EVERY_N_FRAMES = 90

# Defect output
defects_folder       = "defects"      # Directory for defect crop images
defects_log_csv      = "defects_log.csv"  # Append-mode CSV defect log

# ── Inspection thresholds ────────────────────────────────────────────────────
SSIM_FAIL_THRESHOLD     = 0.92  # SSIM score below this value → FAIL
DEFECT_PIXEL_RATIO      = 0.05  # Fraction of pixels with |diff|>threshold → FAIL
DIFF_BINARIZE_THR       = 30    # Absolute pixel diff value treated as defective
TMATCH_WARN_THRESHOLD   = 0.80  # Template-match score below this → warning
MATCH_THRESHOLD         = 0.25  # Minimum template-match score to accept a label detection

# ── Display ──────────────────────────────────────────────────────────────────
PANEL_W = 640   # Each of the 4 display panels is this wide (px)
PANEL_H = 360   # and this tall (px)

# ── Optical-flow strip (software encoder) ────────────────────────────────────
# Width (px) of the centre column used for vertical movement estimation.
# Narrower = faster computation; wider = more stable estimate.
FLOW_STRIP_PX = 40


# ═════════════════════════════════════════════════════════════════════════════
#  DERIVED CONSTANTS  — computed from user config; do not edit directly
# ═════════════════════════════════════════════════════════════════════════════
LABEL_W_PX = int(label_width_cm  * camera_pixels_per_cm)
LABEL_H_PX = int(label_height_cm * camera_pixels_per_cm)
ROI_TOP_PX  = int(roi_top_offset_cm * camera_pixels_per_cm)


# ═════════════════════════════════════════════════════════════════════════════
#  LOGGING SETUP
# ═════════════════════════════════════════════════════════════════════════════
os.makedirs(defects_folder, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("inspection.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("label_inspector")

log.info(
    f"Label size: {LABEL_W_PX}×{LABEL_H_PX} px  "
    f"({label_width_cm}×{label_height_cm} cm @ {camera_pixels_per_cm} px/cm)"
)

# =============================================================================
#  CAMERA CAPTURE THREAD
# =============================================================================

class CameraThread(threading.Thread):
    """
    Daemon thread that reads frames from a USB camera as fast as the hardware
    allows and places them in a bounded queue (maxsize=2).

    When the queue is full the *oldest* frame is discarded so that the
    processing side always receives the freshest image.  This prevents an
    ever-growing lag at variable web speeds.
    """

    def __init__(self, index: "int | str", frame_queue: "queue.Queue[np.ndarray]"):
        super().__init__(daemon=True, name="CameraThread")
        self.index       = index
        self.queue       = frame_queue
        self._stop_evt      = threading.Event()
        self.error: str    = ""
        self.stopped_cleanly: bool = False

    # ------------------------------------------------------------------
    def run(self) -> None:
        cap = self._open_camera()
        if cap is None:
            return  # error already logged in _open_camera

        is_file = isinstance(self.index, str)
        # Throttle file playback to the video's native FPS so the optical-flow
        # accumulator sees the same pixel-per-second motion as a live camera.
        if is_file:
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_delay = 1.0 / native_fps if native_fps > 0 else 1.0 / 30.0
        else:
            frame_delay = 0.0

        while not self._stop_evt.is_set():
            t_frame_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                if is_file:
                    if VIDEO_LOOP:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        log.info("Video ended — looping.")
                        continue
                    else:
                        log.info("Video ended — stopping.")
                        self.stopped_cleanly = True
                        break
                log.warning("Camera read() returned False — retrying in 10 ms")
                time.sleep(0.01)
                continue

            # No resize — use native resolution for best comparison quality
            # For a sideways-mounted camera: uncomment rotate + resize to (240, 360)
            # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # frame = cv2.resize(frame, (240, 360))

            if is_file:
                # For file sources: block until queue has space so no frames are
                # skipped — optical flow needs consecutive frames to detect motion.
                self.queue.put(frame)
            else:
                # For live cameras: drop stale frames to always deliver the freshest.
                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        pass
                self.queue.put(frame)

            # Throttle to native video FPS for file sources
            if frame_delay > 0:
                elapsed = time.perf_counter() - t_frame_start
                sleep_t = frame_delay - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        cap.release()
        log.info("Camera released.")

    # ------------------------------------------------------------------
    def _open_camera(self) -> "cv2.VideoCapture | None":
        """
        Open a USB camera (integer index) or a video file (string path).
        For file sources the backend loop and resolution override are skipped
        — the video's native resolution is used as-is.
        """
        is_file = isinstance(self.index, str)

        if is_file:
            if not os.path.isfile(self.index):
                self.error = f"Video file not found: '{self.index}'"
                log.error(self.error)
                return None
            cap = cv2.VideoCapture(self.index)
            if not cap.isOpened():
                self.error = f"Cannot open video file: '{self.index}'"
                log.error(self.error)
                return None
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_v = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            log.info(
                f"Video file '{self.index}' opened: {w}x{h} px, "
                f"{fps_v:.1f} fps, {frames} frames, loop={VIDEO_LOOP}"
            )
            return cap

        # --- USB camera: try native backend first, fall back to generic ----
        for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
            cap = cv2.VideoCapture(self.index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH_PX)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT_PX)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimise internal buffer lag

            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            log.info(
                f"Camera {self.index} opened at {actual_w}x{actual_h} px "
                f"(backend={backend})"
            )
            return cap

        self.error = f"Cannot open camera index {self.index}"
        log.error(self.error)
        return None

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop_evt.set()


# =============================================================================
#  SOFTWARE ENCODER  (optical-flow vertical movement accumulator)
# =============================================================================

class SoftwareEncoder:
    """
    Estimates cumulative vertical web movement from consecutive frames using
    Gunnar Farneback dense optical flow computed on a narrow centre-column
    strip.  When the accumulated displacement reaches LABEL_H_PX pixels the
    encoder fires a trigger, signalling that one full label pitch has passed
    the camera lens.

    ┌─ FUTURE HARDWARE ENCODER INTEGRATION ──────────────────────────────┐
    │  Most label presses expose encoder pulses via:                      │
    │    • GPIO / quadrature counter (Raspberry Pi, NI-DAQ, Arduino)      │
    │    • Modbus TCP register  (Allen-Bradley, Siemens PLCs)             │
    │    • USB-HID quadrature dongle                                      │
    │                                                                     │
    │  Conversion formula:                                                │
    │    px_per_tick = camera_pixels_per_cm / (10.0 * encoder_ppr_per_mm)│
    │                                                                     │
    │  To integrate:                                                      │
    │    1. Replace the optical-flow block in update() with:              │
    │         ticks = read_encoder_delta()                                │
    │         vy    = ticks * px_per_tick                                 │
    │    2. Keep the accumulator and trigger logic identical.             │
    │  The rest of the pipeline (crop, compare, display) is unchanged.   │
    └─────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, label_height_px: int):
        self._target     = float(label_height_px)
        self._acc        = 0.0
        self._prev_strip: "np.ndarray | None" = None

    # ------------------------------------------------------------------
    def update(self, frame: np.ndarray) -> tuple:
        """
        Feed the latest camera frame.

        Returns:
            triggered (bool) : True when a full label pitch has passed.
            accumulated_px (float) : Current accumulator value (0 … target).
        """
        # Downscale full frame for flow — captures motion anywhere, not just centre strip
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (80, 120), interpolation=cv2.INTER_AREA)
        strip = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        flow_scale = h / 120.0  # scale flow back to native pixel coordinates

        # First frame — just store and return
        if self._prev_strip is None or self._prev_strip.shape != strip.shape:
            self._prev_strip = strip
            return False, self._acc

        try:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_strip, strip,
                flow=None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            # Vertical component scaled back to native pixel coordinates
            vy = float(np.median(np.abs(flow[..., 1]))) * flow_scale
        except cv2.error as exc:
            log.debug(f"Optical flow skipped: {exc}")
            vy = 0.0

        self._prev_strip  = strip
        self._acc        += vy
        log.debug(f"Encoder: vy={vy:.3f}  acc={self._acc:.1f}/{self._target:.1f}")

        if self._acc >= self._target:
            self._acc -= self._target   # carry fractional pixels to next label
            return True, self._acc

        return False, self._acc

    # ------------------------------------------------------------------
    @property
    def progress(self) -> float:
        """Fraction of the way to the next label trigger (0.0 – 1.0)."""
        return min(self._acc / self._target, 1.0) if self._target > 0 else 0.0


# =============================================================================
#  LABEL CROP & SKEW CORRECTION
# =============================================================================

def crop_labels(frame: np.ndarray, ref_gray: np.ndarray) -> "dict[str, np.ndarray]":
    """
    Locate up to two label positions (above, below) in *frame* via template
    matching + NMS, identical to the approach in learn_labels.py.

    Returns a dict with keys 'above' and/or 'below' mapping to BGR crops.
    Missing positions are absent from the dict.
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
        suppress_y0 = max(0, y - LABEL_H_PX // 2)
        suppress_y1 = min(result_work.shape[0], y + LABEL_H_PX // 2)
        result_work[suppress_y0:suppress_y1, :] = -1.0

    if not detections:
        return {}

    detections_sorted = sorted(detections, key=lambda d: d[1])
    fh = frame.shape[0]
    crops = {}
    for rank, (x, y, _) in enumerate(detections_sorted):
        if len(detections_sorted) == 1:
            position = "above" if y < fh // 2 else "below"
        else:
            position = "above" if rank == 0 else "below"
        crop = frame[y:y + LABEL_H_PX, x:x + LABEL_W_PX]
        if crop.size == 0:
            continue
        if crop.shape[0] < LABEL_H_PX or crop.shape[1] < LABEL_W_PX:
            padded = np.zeros((LABEL_H_PX, LABEL_W_PX, 3), dtype=np.uint8)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded
        crops[position] = correct_skew(crop, max_angle_deg=5.0)
    return crops


def correct_skew(crop: np.ndarray, max_angle_deg: float = 5.0) -> np.ndarray:
    """
    Detect and correct minor label skew (up to ±max_angle_deg) using
    Canny edge detection followed by Probabilistic Hough line fitting.

    Strategy:
      1. Find near-horizontal lines in the crop.
      2. Compute their median angle.
      3. Apply an affine rotation around the crop centre.

    If no reliable angle is detected the original crop is returned unchanged
    — this prevents destructive over-correction on blank or low-contrast labels.
    """
    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    min_line_len = max(1, LABEL_W_PX // 4)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_line_len,
        maxLineGap=20,
    )

    if lines is None or len(lines) == 0:
        return crop

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        if dx == 0:
            continue
        angle_deg = float(np.degrees(np.arctan2(y2 - y1, dx)))
        if abs(angle_deg) <= max_angle_deg:
            angles.append(angle_deg)

    if not angles:
        return crop

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.3:   # sub-pixel — rotation not worth it
        return crop

    h, w = crop.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), median_angle, scale=1.0)
    corrected = cv2.warpAffine(
        crop, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    log.debug(f"Skew corrected: {median_angle:.2f} deg")
    return corrected


# =============================================================================
#  REFERENCE IMAGE MANAGEMENT
# =============================================================================

def load_reference(path: str) -> np.ndarray:
    """
    Load the golden reference label image and resize it to LABEL_W_PX x LABEL_H_PX.

    If the file does not exist or cannot be decoded, a blank (black) placeholder
    is returned so the program starts without crashing.  The operator should
    press 's' over a known-good label to capture and save a proper reference.
    """
    if os.path.isfile(path):
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (LABEL_W_PX, LABEL_H_PX),
                             interpolation=cv2.INTER_AREA)
            log.info(f"Reference loaded: '{path}' -> {LABEL_W_PX}x{LABEL_H_PX} px")
            return img
        log.error(f"cv2.imread failed for '{path}'. Using blank reference.")
    else:
        log.warning(
            f"Reference not found: '{path}'. "
            "Run over a good label and press 's' to save one."
        )
    return np.zeros((LABEL_H_PX, LABEL_W_PX, 3), dtype=np.uint8)


def save_reference(image: np.ndarray, path: str) -> None:
    """Overwrite the golden reference image on disk."""
    cv2.imwrite(path, image)
    log.info(f"New golden reference saved to '{path}'.")


# =============================================================================
#  SSIM HELPER
# =============================================================================

def compute_ssim(
    img_a: np.ndarray,
    img_b: np.ndarray,
) -> "tuple[float, np.ndarray]":
    """
    Compute Structural Similarity Index (SSIM) between two BGR images.

    Returns:
        score     : float in [-1, 1]; 1.0 = identical.
        ssim_vis  : uint8 greyscale map of local SSIM values (for display).

    Falls back to Normalised Cross-Correlation (NCC) if scikit-image is
    unavailable — NCC is less sensitive to luminance shifts but acceptable
    as a fallback.
    """
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    if SKIMAGE_OK:
        score, ssim_map = skimage_ssim(
            gray_a, gray_b,
            full=True,
            data_range=255,
        )
        # ssim_map is in [-1, 1]; map to [0, 255] for visualisation
        ssim_vis = np.clip((ssim_map + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)
    else:
        # NCC fallback — single global score, uniform map
        res   = cv2.matchTemplate(gray_a, gray_b, cv2.TM_CCOEFF_NORMED)
        score = float(res[0, 0])
        gray_val = int(np.clip((score + 1.0) / 2.0 * 255, 0, 255))
        ssim_vis  = np.full_like(gray_a, gray_val, dtype=np.uint8)

    return float(score), ssim_vis


# =============================================================================
#  COMPARISON ENGINE
# =============================================================================

def compare_labels(
    crop: np.ndarray,
    reference: np.ndarray,
) -> "tuple[bool, float, float, float, np.ndarray, np.ndarray]":
    """
    Compare a live label crop against the golden reference using three metrics.

    Metrics
    -------
    1. Absolute-difference defect ratio
       Binarise |crop_gray - ref_gray| at DIFF_BINARIZE_THR.
       Ratio = defective_pixels / total_pixels.
       Fast, pixel-perfect, sensitive to registration errors.

    2. SSIM score (structural similarity)
       Perceptually-weighted; tolerates small lighting shifts better than
       raw abs-diff.  Uses scikit-image if available, NCC fallback otherwise.

    3. Template-match score (TM_CCOEFF_NORMED)
       Good secondary indicator; robust to global brightness changes.
       Used only as a warning flag, not the primary PASS/FAIL gate.

    Returns
    -------
    passed       : True if label meets all thresholds
    ssim_score   : float in [-1, 1]
    defect_ratio : fraction of pixels flagged as defective
    tmatch_score : normalised cross-correlation score
    diff_heatmap : BGR heatmap of the absolute difference (for display)
    ssim_vis     : uint8 greyscale SSIM map (for display)
    """
    # Ensure both images are the same size (guard against miscalibration)
    ref_resized = cv2.resize(reference, (crop.shape[1], crop.shape[0]),
                             interpolation=cv2.INTER_AREA)

    gray_crop = cv2.cvtColor(crop,        cv2.COLOR_BGR2GRAY)
    gray_ref  = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2GRAY)

    # --- Metric 1: absolute-difference defect ratio ----------------------
    diff_gray = cv2.absdiff(gray_crop, gray_ref)
    _, diff_bin = cv2.threshold(
        diff_gray, DIFF_BINARIZE_THR, 255, cv2.THRESH_BINARY
    )
    defect_ratio = float(np.count_nonzero(diff_bin)) / diff_bin.size

    # Heatmap: apply JET colormap to the raw diff for intuitive visualisation
    diff_norm    = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
    diff_heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    # --- Metric 2: SSIM --------------------------------------------------
    ssim_score, ssim_vis = compute_ssim(crop, ref_resized)

    # --- Metric 3: template match ----------------------------------------
    res          = cv2.matchTemplate(gray_crop, gray_ref, cv2.TM_CCOEFF_NORMED)
    tmatch_score = float(res[0, 0])

    # --- PASS / FAIL decision --------------------------------------------
    ssim_ok   = ssim_score   >= SSIM_FAIL_THRESHOLD
    defect_ok = defect_ratio <= DEFECT_PIXEL_RATIO
    passed    = ssim_ok and defect_ok

    if not passed:
        log.info(
            f"FAIL — SSIM={ssim_score:.4f} "
            f"defect_ratio={defect_ratio:.3f} "
            f"tmatch={tmatch_score:.4f}"
        )
    elif tmatch_score < TMATCH_WARN_THRESHOLD:
        log.warning(
            f"PASS but low template-match score ({tmatch_score:.4f}) — "
            "possible colour shift or lighting change."
        )

    return passed, ssim_score, defect_ratio, tmatch_score, diff_heatmap, ssim_vis


# =============================================================================
#  FRAME ANNOTATION
# =============================================================================

def draw_roi_on_frame(
    frame: np.ndarray,
    detections: "dict[str, tuple[int,int,bool]]",  # pos -> (x, y, passed)
) -> np.ndarray:
    """
    Draw one bounding box per detected label position on the full frame.
    Green = PASS, Red = FAIL.  Returns a copy so the original is untouched.
    """
    annotated = frame.copy()
    for position, (x, y, passed) in detections.items():
        colour = (0, 220, 0) if passed else (0, 0, 220)
        cv2.rectangle(annotated,
                      (x, y), (x + LABEL_W_PX, y + LABEL_H_PX),
                      colour, thickness=3)
        cv2.putText(annotated, position,
                    (x + 4, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return annotated


# =============================================================================
#  2x2 COMPOSITE DISPLAY
# =============================================================================

def build_display(
    frame:         np.ndarray,
    label_results: "dict[str, dict]",  # pos -> {crop, diff_hmap, ssim_vis, passed, ssim_score, defect_ratio}
    detections_xy: "dict[str, tuple[int,int,bool]]",  # pos -> (x, y, passed) for ROI boxes
    fps:           float,
    stats:         dict,  # session statistics for the status panel
    reference:     "np.ndarray | None" = None,  # golden reference image for panel D
) -> np.ndarray:
    """
    Build a display grid:

        [ Full frame + ROI boxes  |  above crop       |  below crop  ]
        [ Session stats           |  above diff hmap  |  above SSIM  ]

    Layout: 3 columns x 2 rows.  Each panel is PANEL_W x PANEL_H.
    """
    pw, ph = PANEL_W, PANEL_H

    def _resize(img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)

    def _placeholder(text: str) -> np.ndarray:
        p = np.zeros((ph, pw, 3), dtype=np.uint8)
        cv2.putText(p, text, (pw // 2 - 80, ph // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        return p

    overall_passed = all(r["passed"] for r in label_results.values()) if label_results else True

    # --- Panel A: reference image -------------------------------------------
    status_text  = "PASS" if overall_passed else "FAIL"
    status_color = (0, 220, 0) if overall_passed else (0, 0, 220)
    panel_a = np.zeros((ph, pw, 3), dtype=np.uint8)
    panel_a[:] = (25, 25, 25)
    if reference is not None:
        rh_a, rw_a = reference.shape[:2]
        scale_a = min(pw / rw_a, ph / rh_a)
        tw_a, th_a = int(rw_a * scale_a), int(rh_a * scale_a)
        ref_full = cv2.resize(reference, (tw_a, th_a), interpolation=cv2.INTER_AREA)
        x_a = (pw - tw_a) // 2
        y_a = (ph - th_a) // 2
        panel_a[y_a:y_a + th_a, x_a:x_a + tw_a] = ref_full
    cv2.putText(panel_a, "A", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(panel_a, "Reference", (28, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)

    _crop_panel_labels = {"above": "B", "below": "C"}
    def _crop_panel(pos: str) -> np.ndarray:
        r = label_results.get(pos)
        if r is None:
            return _placeholder(f"{pos} (none)")
        p = _resize(r["crop"])
        p_txt = "PASS" if r["passed"] else "FAIL"
        p_col = (0, 220, 0) if r["passed"] else (0, 0, 220)
        cv2.putText(p, _crop_panel_labels.get(pos, ""), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(p, f"{pos}  SSIM={r['ssim_score']:.4f}",
                    (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
        cv2.putText(p, p_txt, (8, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, p_col, 2)
        return p

    def _diff_panel(pos: str) -> np.ndarray:
        r = label_results.get(pos)
        if r is None:
            return _placeholder(f"{pos} diff")
        p = _resize(r["diff_hmap"])
        cv2.putText(p, f"{pos} defect={r['defect_ratio']*100:.1f}%",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
        return p


    # --- Panel D: live frame with overlays ----------------------------------
    panel_d = _resize(draw_roi_on_frame(frame, detections_xy))
    cv2.putText(panel_d, "D", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(panel_d, status_text,
                (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1.4, status_color, 3)
    cv2.putText(panel_d, f"FPS: {fps:.1f}",
                (10, ph - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 0), 2)
    lines = [
        ("Session Status", (0, 200, 200), 0.6, 2),
        (f"Frames : {stats.get('frames', 0)}",       (200, 200, 200), 0.5, 1),
        (f"Both   : {stats.get('both', 0)}",          (200, 200, 200), 0.5, 1),
        (f"A PASS : {stats.get('above_pass_pct', 0.0):.1f}%", (0, 220, 0), 0.5, 1),
        (f"B PASS : {stats.get('below_pass_pct', 0.0):.1f}%", (0, 220, 0), 0.5, 1),
        (f"Defects: {stats.get('defects', 0)}",      (0, 80, 220), 0.5, 1),
    ]
    for i, (text, color, scale, thickness) in enumerate(lines):
        cv2.putText(panel_d, text,
                    (pw - 180, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    # --- Stitch 3x2 grid -------------------------------------------------
    top_row    = np.hstack([panel_a, _crop_panel("above"), _crop_panel("below")])
    bottom_row = np.hstack([panel_d, _diff_panel("above"), _diff_panel("below")])
    return np.vstack([top_row, bottom_row])


# =============================================================================
#  DEFECT SAVING & LOGGING
# =============================================================================

def save_defect(
    crop:        np.ndarray,
    diff_hmap:   np.ndarray,
    label_count: int,
    frame_no:    int = 0,
    position:    str = "",
) -> str:
    """
    Save the defect crop and its diff heatmap to the defects folder.
    Filenames include timestamp, frame number, position, and label counter.
    Returns the base filename used (without extension) for CSV logging.
    """
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms precision
    pos_tag  = f"_{position}" if position else ""
    basename = f"defect_{ts}_frame{frame_no:06d}{pos_tag}_label{label_count:06d}"

    crop_path = os.path.join(defects_folder, basename + "_crop.png")
    diff_path = os.path.join(defects_folder, basename + "_diff.png")

    cv2.imwrite(crop_path, crop)
    cv2.imwrite(diff_path, diff_hmap)
    log.info(f"Defect saved: {crop_path}")
    return basename


def log_defect_csv(
    basename:     str,
    ssim_score:   float,
    defect_ratio: float,
    tmatch_score: float,
    label_count:  int,
) -> None:
    """
    Append one row to the defects CSV log.
    Creates the file with a header if it does not exist yet.
    CSV columns: timestamp, label_number, ssim, defect_pct, tmatch, filename
    """
    file_exists = os.path.isfile(defects_log_csv)
    with open(defects_log_csv, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow(
                ["timestamp", "label_number",
                 "ssim", "defect_pct", "tmatch_score", "filename"]
            )
        writer.writerow([
            datetime.now().isoformat(timespec="milliseconds"),
            label_count,
            f"{ssim_score:.6f}",
            f"{defect_ratio * 100:.3f}",
            f"{tmatch_score:.6f}",
            basename,
        ])


# =============================================================================
#  MAIN
# =============================================================================

def main() -> None:
    """
    Application entry point.

    Event loop
    ----------
    1. Pull the latest frame from the camera queue (non-blocking).
    2. Feed the frame to the SoftwareEncoder to accumulate vertical movement.
    3. When the encoder fires a trigger, crop and inspect the label.
    4. Build and show the 2x2 composite display every frame.
    5. Handle key presses ('q' = quit, 's' = save reference).
    """
    log.info("=" * 60)
    log.info("Label Inspector starting up")
    log.info(f"  SSIM threshold   : {SSIM_FAIL_THRESHOLD}")
    log.info(f"  Defect px ratio  : {DEFECT_PIXEL_RATIO}")
    log.info(f"  Diff binarise thr: {DIFF_BINARIZE_THR}")
    log.info("=" * 60)

    # --- Initialise camera thread ----------------------------------------
    frame_queue   = queue.Queue(maxsize=2)
    cam_thread    = CameraThread(CAMERA_INDEX, frame_queue)
    cam_thread.start()

    # Give the source a moment to open (USB cameras need ~1 s; files are instant)
    time.sleep(0.2 if isinstance(CAMERA_INDEX, str) else 1.0)
    if cam_thread.error:
        log.error(f"Startup failed: {cam_thread.error}")
        log.error("Check CAMERA_INDEX and USB connection, then restart.")
        return

    # --- Load golden reference -------------------------------------------
    reference = load_reference(reference_image_path)
    ref_gray  = cv2.cvtColor(
        cv2.resize(reference, (LABEL_W_PX, LABEL_H_PX), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY,
    )

    # --- State variables -------------------------------------------------
    last_label_results:  "dict[str, dict]" = {}   # pos -> inspection result dict
    last_detections_xy:  "dict[str, tuple]" = {}  # pos -> (x, y, passed)
    last_crop:           "np.ndarray | None" = None  # above crop (for 's' save)
    label_count:         int   = 0
    defect_count:        int   = 0
    frame_no:            int   = 0
    # Per-position pass/fail counters for stats panel
    stats_frames:        int   = 0   # frames where any label detected
    stats_both:          int   = 0
    stats_above_only:    int   = 0
    stats_below_only:    int   = 0
    stats_above_pass:    int   = 0
    stats_above_total:   int   = 0
    stats_below_pass:    int   = 0
    stats_below_total:   int   = 0

    # --- FPS counter (exponential moving average) ------------------------
    fps:          float = 0.0
    fps_alpha:    float = 0.1          # smoothing factor
    t_prev:       float = time.perf_counter()

    log.info("Running — press 'q' to quit, 's' to save current crop as reference.")

    # --- Detect physical screen resolution to size panels correctly --------
    try:
        import subprocess as _sp, re as _re
        _out = _sp.check_output(["system_profiler", "SPDisplaysDataType"], text=True)
        _m = _re.search(r"Resolution:\s*(\d+)\s*x\s*(\d+)", _out)
        if _m:
            _screen_w, _screen_h = int(_m.group(1)), int(_m.group(2))
            global PANEL_W, PANEL_H
            PANEL_W = _screen_w // 3
            PANEL_H = _screen_h // 2
            log.info(f"Screen {_screen_w}x{_screen_h} -> panel {PANEL_W}x{PANEL_H}")
    except Exception:
        pass

    cv2.namedWindow("Label Inspector", cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.setWindowProperty("Label Inspector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        # ── Grab latest frame ────────────────────────────────────────────
        try:
            frame = frame_queue.get(timeout=0.5)
        except queue.Empty:
            if not cam_thread.is_alive():
                if cam_thread.stopped_cleanly:
                    log.info("Video ended. Exiting.")
                else:
                    log.error("Camera thread died unexpectedly. Exiting.")
                break
            log.warning("No frame received for 500 ms — camera stall?")
            continue

        # ── FPS update ───────────────────────────────────────────────────
        t_now  = time.perf_counter()
        dt     = t_now - t_prev
        t_prev = t_now
        if dt > 0:
            fps = (1.0 - fps_alpha) * fps + fps_alpha * (1.0 / dt)

        frame_no += 1

        # ── Per-frame template matching (learn_labels.py approach) ──────
        crops = crop_labels(frame, ref_gray)

        if crops:
            label_count += 1
            frame_results = {}
            frame_det_xy  = {}

            # Recover (x, y) locations from a second template match pass
            _gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _res  = cv2.matchTemplate(_gray, ref_gray, cv2.TM_CCOEFF_NORMED)
            _work = _res.copy()
            _locs = {}
            for _ in range(2):
                _, mv, _, ml = cv2.minMaxLoc(_work)
                if mv < MATCH_THRESHOLD:
                    break
                _locs[ml[1]] = ml
                sy0 = max(0, ml[1] - LABEL_H_PX // 2)
                sy1 = min(_work.shape[0], ml[1] + LABEL_H_PX // 2)
                _work[sy0:sy1, :] = -1.0
            _sorted_locs = [_locs[k] for k in sorted(_locs)]

            for rank, (pos, crop) in enumerate(
                sorted(crops.items(), key=lambda kv: kv[0])  # above < below
            ):
                (
                    passed, ssim_score, defect_ratio, tmatch_score,
                    diff_hmap, ssim_vis,
                ) = compare_labels(crop, reference)

                frame_results[pos] = dict(
                    crop=crop, diff_hmap=diff_hmap, ssim_vis=ssim_vis,
                    passed=passed, ssim_score=ssim_score,
                    defect_ratio=defect_ratio,
                )
                xy = _sorted_locs[rank] if rank < len(_sorted_locs) else (0, 0)
                frame_det_xy[pos] = (xy[0], xy[1], passed)

                if not passed:
                    defect_count += 1
                    basename = save_defect(crop, diff_hmap, label_count,
                                               frame_no=frame_no, position=pos)
                    log_defect_csv(basename, ssim_score, defect_ratio,
                                   tmatch_score, label_count)

                log.info(
                    f"Label #{label_count:06d} {pos} | "
                    f"{'PASS' if passed else 'FAIL'} | "
                    f"SSIM={ssim_score:.4f} | "
                    f"defect={defect_ratio*100:.2f}% | "
                    f"tmatch={tmatch_score:.4f} | "
                    f"FPS={fps:.1f} | "
                    f"total_defects={defect_count}"
                )

            # Update session stats
            stats_frames += 1
            has_above = "above" in frame_results
            has_below = "below" in frame_results
            if has_above and has_below:
                stats_both += 1
            elif has_above:
                stats_above_only += 1
            elif has_below:
                stats_below_only += 1
            if has_above:
                stats_above_total += 1
                if frame_results["above"]["passed"]:
                    stats_above_pass += 1
            if has_below:
                stats_below_total += 1
                if frame_results["below"]["passed"]:
                    stats_below_pass += 1

            last_label_results = frame_results
            last_detections_xy = frame_det_xy
            last_crop = crops.get("above", next(iter(crops.values())))

        # ── Build and show composite display ────────────────────────────
        display = build_display(
            frame         = frame,
            label_results = last_label_results,
            detections_xy = last_detections_xy,
            fps           = fps,
            reference     = reference,
            stats         = dict(
                frames         = stats_frames,
                both           = stats_both,
                above_only     = stats_above_only,
                below_only     = stats_below_only,
                above_pass_pct = (stats_above_pass / stats_above_total * 100) if stats_above_total else 0.0,
                below_pass_pct = (stats_below_pass / stats_below_total * 100) if stats_below_total else 0.0,
                defects        = defect_count,
            ),
        )
        cv2.imshow("Label Inspector", display)

        # ── Key handler ──────────────────────────────────────────────────
        # In step mode, block after a label detection until a key is pressed;
        # otherwise poll with a 1 ms timeout so the display stays responsive.
        if DEBUG_STEP_MODE and crops:
            log.info("[DEBUG] Paused — press any key to continue, 'q' to quit.")
            key = cv2.waitKey(0) & 0xFF
        else:
            key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            log.info("Quit key pressed.")
            break

        elif key == ord("s"):
            if last_crop is not None:
                save_reference(last_crop, reference_image_path)
                reference = load_reference(reference_image_path)
                ref_gray  = cv2.cvtColor(
                    cv2.resize(reference, (LABEL_W_PX, LABEL_H_PX), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2GRAY,
                )
                log.info("Reference updated from current crop.")
            else:
                log.warning("No crop available yet — wait for the first label trigger.")

    # ── Shutdown ─────────────────────────────────────────────────────────
    cam_thread.stop()
    cam_thread.join(timeout=3.0)
    cv2.destroyAllWindows()
    log.info(
        f"Session ended. Labels inspected: {label_count} | "
        f"Defects found: {defect_count}"
    )


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
