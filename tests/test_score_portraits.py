import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from score_portraits import (
    MIN_EYE_WIDTH,
    _eye_bbox,
    _score_eye,
    score_image,
    whole_image_sharpness,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _make_landmark(x, y, visibility=1.0):
    # Mimics a MediaPipe NormalizedLandmark.
    # x, y: normalized [0.0, 1.0] coordinates relative to image dimensions.
    # visibility: confidence the landmark is visible; <0.5 causes it to be filtered out.
    return SimpleNamespace(x=x, y=y, visibility=visibility)


def _make_detector(face_landmarks=None):
    # Returns a mock MediaPipe FaceLandmarker that bypasses the 26MB model.
    # face_landmarks: list of landmark lists (one per detected face).
    # Empty list = no faces detected — triggers fallback path in score_image.
    result = SimpleNamespace(face_landmarks=face_landmarks or [])
    detector = MagicMock()
    detector.detect.return_value = result
    return detector


# ---------------------------------------------------------------------------
# whole_image_sharpness
# ---------------------------------------------------------------------------

def test_whole_image_sharpness_sharp_gt_blurry():
    # Validates the core Laplacian variance assumption: high-contrast edges produce
    # higher scores than blurred images. If this fails, the entire scoring pipeline
    # is unreliable regardless of face detection.
    # sharp.jpg: 64x64 high-contrast vertical grid lines.
    # blurry.jpg: same grid with radius-8 Gaussian blur applied.
    import cv2
    sharp = cv2.imread(str(FIXTURES / "sharp.jpg"))
    blurry = cv2.imread(str(FIXTURES / "blurry.jpg"))
    assert whole_image_sharpness(sharp) > whole_image_sharpness(blurry)


# ---------------------------------------------------------------------------
# _eye_bbox
# ---------------------------------------------------------------------------

def test_eye_bbox_returns_none_too_few_visible():
    # _eye_bbox requires at least 4 visible landmarks to compute a meaningful bbox.
    # Fewer than 4 means the eye is too occluded to score — returns None to trigger
    # the fallback path. visibility=0.1 is below VISIBILITY_MIN (0.5), so all
    # landmarks fail the filter, resulting in 0 visible points.
    lms = [_make_landmark(0.5, 0.5, 0.1) for _ in range(478)]
    from score_portraits import LEFT_EYE
    result = _eye_bbox(lms, LEFT_EYE, 100, 100)
    assert result is None


def test_eye_bbox_clamped_to_image():
    # When eye landmarks are near the image edge, the 20% padding (EYE_PAD)
    # can push the bbox outside image boundaries. The clamp (max(0,...), min(img_w,...))
    # must prevent negative coords or out-of-bounds crops, which would produce
    # empty arrays and crash cv2.cvtColor.
    # Landmarks at (0.98, 0.98) in a 100x100 image = pixel (98, 98) — right at the edge.
    lms = [_make_landmark(0.98, 0.98, 1.0) for _ in range(478)]
    from score_portraits import LEFT_EYE
    bbox = _eye_bbox(lms, LEFT_EYE, 100, 100)
    if bbox is not None:
        x, y, w, h = bbox
        assert x >= 0 and y >= 0
        assert x + w <= 100
        assert y + h <= 100


# ---------------------------------------------------------------------------
# _score_eye
# ---------------------------------------------------------------------------

def test_score_eye_too_narrow():
    # MIN_EYE_WIDTH (20px) is the minimum pre-resize eye crop width we'll score.
    # Below this, upscaling to 128x128 can't recover sharpness that isn't there —
    # the score would be noise. Returns None to signal "not scoreable".
    # bbox width here is MIN_EYE_WIDTH - 1, just below the threshold.
    import numpy as np
    bgr = np.zeros((100, 100, 3), dtype="uint8")
    result = _score_eye(bgr, (0, 0, MIN_EYE_WIDTH - 1, 20))
    assert result is None


def test_score_eye_valid():
    # Confirms _score_eye returns a positive float for a scoreable crop.
    # Uses random-noise image (high Laplacian variance by nature) with a bbox
    # wide enough to pass the MIN_EYE_WIDTH gate (40px > 20px).
    import numpy as np
    bgr = (np.random.randint(0, 256, (100, 100, 3))).astype("uint8")
    result = _score_eye(bgr, (10, 10, 40, 20))
    assert result is not None
    assert result > 0


# ---------------------------------------------------------------------------
# score_image
# ---------------------------------------------------------------------------

def test_score_image_no_face():
    # When face detection returns no landmarks, score_image must:
    #   1. Set face_count=0
    #   2. Set fallback_used=True (signals to agent this photo needs manual review)
    #   3. Leave eye_sharpness_min and eye_sharpness_max empty (not scoreable)
    # This is the expected path for environmental shots, backs-of-heads, etc.
    detector = _make_detector(face_landmarks=[])
    row = score_image(FIXTURES / "sharp.jpg", detector)
    assert row is not None
    assert row["face_count"] == 0
    assert row["fallback_used"] is True
    assert row["eye_sharpness_min"] == ""
    assert row["eye_sharpness_max"] == ""


def test_score_image_unreadable(tmp_path):
    # score_image raises OSError (PIL.UnidentifiedImageError) for corrupt/non-image files.
    # Callers (score_batch, main) catch and log; the exception propagates to keep
    # error messages explicit rather than silently returning None.
    import pytest
    bad = tmp_path / "not_an_image.jpg"
    bad.write_text("garbage")
    detector = _make_detector()
    with pytest.raises(OSError):
        score_image(bad, detector)


def test_score_image_whole_sharpness_present():
    # whole_image_sharpness is computed for every row, including fallback cases.
    # It serves as the fallback score when no face is detected and is always
    # available for cross-checking against eye scores via get_metadata.
    detector = _make_detector(face_landmarks=[])
    row = score_image(FIXTURES / "sharp.jpg", detector)
    assert row["whole_image_sharpness"] > 0
