#!/usr/bin/env python3
"""
triage-keepers — Evening 1 prototype

Scores eye-region sharpness for portrait JPGs using MediaPipe Face Landmarker
+ Laplacian variance. No MCP, no cache — throwaway script to validate the CV claim.

Usage:
    python score_portraits.py <folder> [output.csv] [model.task]

Output CSV columns:
    path, face_count, eye_sharpness_min, eye_sharpness_max,
    whole_image_sharpness, fallback_used
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
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def whole_image_sharpness(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (NORM_WHOLE_SIZE, NORM_WHOLE_SIZE), interpolation=cv2.INTER_LINEAR)
    return _laplacian_var(thumb)


def _eye_bbox(face_lms, eye_indices: list[int], img_w: int, img_h: int) -> tuple | None:
    """
    Padded pixel bbox for one eye.  Returns (x, y, w, h) or None if fewer
    than 4 landmarks clear the visibility threshold.
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
    Score one eye region.  Returns None if the pre-resize width is below
    MIN_EYE_WIDTH (upscaling can't recover sharpness that isn't there).
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
# Per-image scoring
# ---------------------------------------------------------------------------

def score_image(path: Path, detector) -> dict | None:
    """
    Score one JPG.  Returns a CSV row dict, or None if the file is unreadable.
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
        return {**base, "eye_sharpness_min": "", "eye_sharpness_max": "", "fallback_used": True}

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
        # Face detected but no eye was scoreable (occlusion, too small, etc.)
        return {**base, "eye_sharpness_min": "", "eye_sharpness_max": "", "fallback_used": True}

    return {
        **base,
        "eye_sharpness_min": round(min(scores), 4),
        "eye_sharpness_max": round(max(scores), 4),
        "fallback_used":     False,
    }


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
]


def main(folder: str, output_csv: str = "scores.csv", model: str = MODEL_FILENAME) -> None:
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

    print(f"{len(jpgs)} JPGs found. Scoring…")

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
