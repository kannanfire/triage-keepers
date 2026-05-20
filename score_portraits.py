#!/usr/bin/env python3
"""
triage-keepers — Evening 1 + 5 prototype

Scores eye-region sharpness and extracts camera metadata for portrait JPGs using
MediaPipe Face Landmarker + Laplacian variance. Validates CV claim and indexes
metadata for caching (no MCP yet in this script, but scores feed into server.py).

Usage:
    python score_portraits.py <folder> [output.csv] [model.task]

Output CSV columns:
    path, face_count, eye_sharpness_min, eye_sharpness_max,
    whole_image_sharpness, fallback_used, iso, shutter, aperture,
    focal_length, camera, taken_at
"""

import csv
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NORM_EYE_SIZE   = 128    # eye crops resized to this square before Laplacian
NORM_WHOLE_SIZE = 512    # whole-image resize for fallback sharpness score
MIN_EYE_WIDTH   = 20    # skip eye bboxes narrower than this (pre-resize px)
EYE_PAD         = 0.20  # fractional padding added around raw eye-contour bbox

MODEL_FILENAME = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# Canonical MediaPipe face-mesh eye-contour landmark indices (out of 478 total).
# Source: MediaPipe canonical face model topology.
LEFT_EYE  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

VISIBILITY_MIN = 0.5  # landmarks below this are treated as not visible

# ---------------------------------------------------------------------------
# CV helpers
# ---------------------------------------------------------------------------

def _laplacian_var(gray: np.ndarray) -> float:
    """
    Compute Laplacian variance on grayscale image.

    Laplacian is a second-derivative edge-detection filter. High-contrast edges
    (sharp focus) produce large variance; smooth/blurred regions produce small.
    This is the core metric for focus quality.

    gray: grayscale numpy array (H x W, single channel)
    Returns: float variance of Laplacian operator applied to gray
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def whole_image_sharpness(bgr: np.ndarray) -> float:
    """
    Compute sharpness score for entire image via normalized Laplacian variance.

    Used as fallback when face detection fails (no eye region to score).
    Normalizes to NORM_WHOLE_SIZE (512x512) so scores are comparable across
    different camera resolutions and zoom levels.

    bgr: BGR image from cv2.imread() (H x W x 3)
    Returns: float Laplacian variance on normalized grayscale
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (NORM_WHOLE_SIZE, NORM_WHOLE_SIZE), interpolation=cv2.INTER_LINEAR)
    return _laplacian_var(thumb)


def _eye_bbox(face_lms, eye_indices: list[int], img_w: int, img_h: int) -> tuple | None:
    """
    Compute padded bounding box for one eye region from MediaPipe landmarks.

    MediaPipe returns normalized [0.0, 1.0] coordinates. This function:
    1. Filters landmarks by visibility (>= VISIBILITY_MIN or 0.0)
    2. Requires at least 4 visible landmarks to form meaningful bbox
    3. Computes axis-aligned bbox from min/max x,y
    4. Adds EYE_PAD (20%) padding on all sides to include context
    5. Clamps to image bounds to prevent out-of-bounds crops

    face_lms: list of 478 MediaPipe landmarks (NormalizedLandmark objects)
    eye_indices: list of 16 landmark indices defining one eye contour
    img_w, img_h: image width and height in pixels
    Returns: (x, y, w, h) in pixel coords, or None if too few visible landmarks
    """
    visible = []
    for idx in eye_indices:
        lm = face_lms[idx]
        # visibility may be 0.0 when the field is unpopulated; treat as visible
        # rather than silently discarding landmarks on models that don't fill it.
        vis = lm.visibility if lm.visibility is not None else 1.0
        if vis >= VISIBILITY_MIN or vis == 0.0:
            visible.append(lm)

    if len(visible) < 4:
        return None

    xs = [int(lm.x * img_w) for lm in visible]
    ys = [int(lm.y * img_h) for lm in visible]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    raw_w = x_max - x_min
    raw_h = y_max - y_min
    pad_x = int(raw_w * EYE_PAD)
    pad_y = int(raw_h * EYE_PAD)

    x  = max(0, x_min - pad_x)
    y  = max(0, y_min - pad_y)
    x2 = min(img_w, x_max + pad_x)
    y2 = min(img_h, y_max + pad_y)

    return (x, y, x2 - x, y2 - y)


def _score_eye(bgr: np.ndarray, bbox: tuple) -> float | None:
    """
    Score sharpness of one eye crop via normalized Laplacian variance.

    Crops the eye region from the full image, converts to grayscale, resizes to
    NORM_EYE_SIZE (128x128) for scale-independent scoring, then computes Laplacian
    variance. Returns None if the crop is too small (w < MIN_EYE_WIDTH) — upscaling
    won't recover sharpness information that isn't present.

    bgr: full BGR image (H x W x 3)
    bbox: (x, y, w, h) pixel coordinates from _eye_bbox()
    Returns: float Laplacian variance, or None if crop too narrow to be meaningful
    """
    x, y, w, h = bbox
    if w < MIN_EYE_WIDTH:
        return None

    crop = bgr[y:y + h, x:x + w]
    if crop.size == 0:
        return None

    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (NORM_EYE_SIZE, NORM_EYE_SIZE), interpolation=cv2.INTER_LINEAR)
    return _laplacian_var(resized)


# ---------------------------------------------------------------------------
# EXIF extraction
# ---------------------------------------------------------------------------

def extract_exif(path: Path) -> dict:
    """
    Extract camera metadata from JPG EXIF tags.

    Pillow's getexif() returns all EXIF IFD tags as a dict keyed by integer tag IDs.
    Extracts specific tags defined by the EXIF standard:
    - 34855 (ISOSpeedRatings): ISO sensitivity (int)
    - 33434 (ExposureTime): shutter speed as Fraction, converted to string "1/250" (str)
    - 33437 (FNumber): aperture as Fraction, converted to float (float)
    - 37386 (FocalLength): focal length in mm as Fraction, converted to float (float)
    - 271 (Model): camera model name (str)
    - 36867 (DateTimeOriginal): shot datetime in "YYYY:MM:DD HH:MM:SS" format (str)

    path: Path to JPG file
    Returns: dict with keys {iso, shutter, aperture, focal_length, camera, taken_at}
             Missing tags are set to None.
    """
    exif_dict = {
        "iso": None,
        "shutter": None,
        "aperture": None,
        "focal_length": None,
        "camera": None,
        "taken_at": None,
    }

    try:
        img = PILImage.open(path)
        exif = img.getexif()

        if exif:
            if 34855 in exif:
                exif_dict["iso"] = int(exif[34855])

            if 33434 in exif:
                frac = exif[33434]
                if hasattr(frac, 'numerator') and hasattr(frac, 'denominator'):
                    exif_dict["shutter"] = f"{frac.numerator}/{frac.denominator}"
                else:
                    exif_dict["shutter"] = str(frac)

            if 33437 in exif:
                frac = exif[33437]
                if hasattr(frac, 'numerator') and hasattr(frac, 'denominator'):
                    exif_dict["aperture"] = float(frac.numerator) / float(frac.denominator)
                else:
                    exif_dict["aperture"] = float(frac)

            if 37386 in exif:
                frac = exif[37386]
                if hasattr(frac, 'numerator') and hasattr(frac, 'denominator'):
                    exif_dict["focal_length"] = float(frac.numerator) / float(frac.denominator)
                else:
                    exif_dict["focal_length"] = float(frac)

            if 271 in exif:
                exif_dict["camera"] = str(exif[271]).strip()

            if 36867 in exif:
                dt_str = str(exif[36867])
                exif_dict["taken_at"] = dt_str

    except Exception as e:
        pass

    return exif_dict


# ---------------------------------------------------------------------------
# Per-image scoring
# ---------------------------------------------------------------------------

def score_image(path: Path, detector) -> dict | None:
    """
    Score one JPG: CV pipeline (face detection, eye sharpness) + EXIF metadata.

    Pipeline:
    1. Load JPG with cv2.imread() → return None if unreadable
    2. Compute whole_image_sharpness (fallback metric)
    3. Run MediaPipe face detection on RGB conversion
    4. If faces detected:
       a. For each face, compute eye bboxes (LEFT_EYE, RIGHT_EYE)
       b. Score each eye region with Laplacian variance
       c. Store min/max scores
    5. If no faces or no scoreable eyes, set fallback_used=True
    6. Extract EXIF metadata (iso, shutter, aperture, focal_length, camera, taken_at)
    7. Merge EXIF into return dict

    path: Path to JPG file
    detector: initialized MediaPipe FaceLandmarker
    Returns: dict with keys {path, face_count, eye_sharpness_min, eye_sharpness_max,
                             whole_image_sharpness, fallback_used, iso, shutter,
                             aperture, focal_length, camera, taken_at}
             or None if file is unreadable
    """
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None

    img_h, img_w = bgr.shape[:2]
    wi_sharp = whole_image_sharpness(bgr)

    rgb      = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)
    n_faces  = len(result.face_landmarks)

    base = {
        "path":                 str(path.resolve()),
        "face_count":           n_faces,
        "whole_image_sharpness": round(wi_sharp, 4),
    }

    if n_faces == 0:
        cv_scores = {"eye_sharpness_min": "", "eye_sharpness_max": "", "fallback_used": True}
    else:
        scores: list[float] = []
        for face_lms in result.face_landmarks:
            for eye_indices in (LEFT_EYE, RIGHT_EYE):
                bbox = _eye_bbox(face_lms, eye_indices, img_w, img_h)
                if bbox is None:
                    continue
                s = _score_eye(bgr, bbox)
                if s is not None:
                    scores.append(s)

        if not scores:
            cv_scores = {"eye_sharpness_min": "", "eye_sharpness_max": "", "fallback_used": True}
        else:
            cv_scores = {
                "eye_sharpness_min": round(min(scores), 4),
                "eye_sharpness_max": round(max(scores), 4),
                "fallback_used":     False,
            }

    exif = extract_exif(path)
    return {**base, **cv_scores, **exif}


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def ensure_model(model_path: Path) -> None:
    if model_path.exists():
        return
    print(f"Downloading Face Landmarker model (~26 MB) → {model_path}")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print("Download complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "path", "face_count",
    "eye_sharpness_min", "eye_sharpness_max",
    "whole_image_sharpness", "fallback_used",
    "iso", "shutter", "aperture", "focal_length", "camera", "taken_at",
]


def main(folder: str, output_csv: str = "scores.csv", model: str = MODEL_FILENAME) -> None:
    """
    Index a folder of JPGs: detect faces, score eye sharpness, extract EXIF.

    Walks folder (recursive) for all JPGs, initializes MediaPipe detector,
    calls score_image() on each file (which includes EXIF extraction),
    and writes results to CSV. Logs errors but continues indexing.

    folder: path to folder (or root of folder tree)
    output_csv: output filename (default "scores.csv")
    model: path to face_landmarker.task (auto-downloaded if missing)
    """
    folder_path = Path(folder).resolve()
    if not folder_path.is_dir():
        sys.exit(f"Not a directory: {folder_path}")

    model_path = Path(model)
    ensure_model(model_path)

    jpgs = sorted(
        p for p in folder_path.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg"}
    )
    if not jpgs:
        sys.exit(f"No JPG/JPEG files found in {folder_path}")

    print(f"{len(jpgs)} JPGs found. Scoring and extracting EXIF…")

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=10,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    errors: list[str] = []

    with mp_vision.FaceLandmarker.create_from_options(options) as detector:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

            for i, path in enumerate(jpgs, 1):
                print(f"\r[{i:>4}/{len(jpgs)}] {path.name:<55}", end="", flush=True)
                try:
                    row = score_image(path, detector)
                    if row is None:
                        errors.append(f"unreadable: {path}")
                    else:
                        writer.writerow(row)
                except Exception as exc:
                    errors.append(f"{path.name}: {exc}")

    scored = len(jpgs) - len(errors)
    print(f"\nDone. {scored}/{len(jpgs)} rows written → {output_csv}")

    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print(f"  {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(f"Usage: python {Path(__file__).name} <folder> [scores.csv] [face_landmarker.task]")
    main(*sys.argv[1:])
