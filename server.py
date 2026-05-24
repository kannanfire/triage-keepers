#!/usr/bin/env python3
"""
triage-keepers MCP server — Evenings 3–6.
Tools: list_folders, get_thumbnail, index_folder, assess_subject_sharpness,
find_unsharp_subjects, find_no_subject, get_metadata.
"""

import io
from pathlib import Path

from mcp.server.fastmcp import FastMCP, Image as MCPImage
from PIL import Image as PILImage
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import cache as _cache
from score_portraits import score_image

mcp = FastMCP("triage-keepers")

_DB_PATH = Path("~/.triage-keepers/cache.db")
_MODEL_PATH = Path("face_landmarker.task")
_conn = None
_detector = None


def _get_conn():
    global _conn
    if _conn is None:
        _conn = _cache.init_db(_DB_PATH)
    return _conn


def _get_detector():
    global _detector
    if _detector is None:
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=10,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        _detector = mp_vision.FaceLandmarker.create_from_options(options)
    return _detector


@mcp.tool()
def list_folders(root: str) -> list[str]:
    """Return immediate subdirectories of root, sorted."""
    p = Path(root).expanduser().resolve()
    if not p.is_dir():
        return []
    return [str(d) for d in sorted(p.iterdir()) if d.is_dir()]


@mcp.tool()
def get_thumbnail(path: str, size: int = 512) -> MCPImage:
    """Return a JPEG thumbnail of path, resized to fit within size×size."""
    img = PILImage.open(path)
    img = img.convert("RGB")
    img.thumbnail((size, size), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return MCPImage(data=buf.getvalue(), format="jpeg")


@mcp.tool()
def index_folder(path: str, recursive: bool = True) -> dict:
    """Walk path for JPGs, score with CV, upsert cache rows. Returns counts."""
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        return {"error": f"Not a directory: {path}"}

    glob = p.rglob("*") if recursive else p.glob("*")
    jpgs = [f for f in glob if f.suffix.lower() in {".jpg", ".jpeg"} and f.is_file()]

    conn = _get_conn()
    detector = _get_detector()
    indexed = 0
    skipped = 0

    for f in jpgs:
        stat = f.stat()
        if _cache.needs_reindex(conn, str(f), stat.st_mtime, stat.st_size):
            row = score_image(f, detector)
            if row is not None:
                row["mtime"] = stat.st_mtime
                row["size"] = stat.st_size
                _cache.upsert_photo(conn, row)
                indexed += 1
            else:
                skipped += 1
        else:
            skipped += 1

    return {"total": len(jpgs), "indexed": indexed, "skipped": skipped}


@mcp.tool()
def assess_subject_sharpness(path: str) -> dict:
    """
    Assess sharpness of one photo's subject (eye region or whole image).

    Looks up the photo in cache. If not cached, scores on-demand using the
    CV pipeline. Returns CV scores and metadata for visual inspection.

    path: absolute path to JPG file
    Returns: dict with face_count, eye_sharpness_min, eye_sharpness_max,
             whole_image_sharpness, fallback_used, camera, taken_at, phash
    """
    conn = _get_conn()
    row = _cache.get_photo(conn, path)

    if row is None:
        detector = _get_detector()
        from pathlib import Path as PathClass
        row = score_image(PathClass(path), detector)
        if row is not None:
            _cache.upsert_photo(conn, row)

    if row is None:
        return {"error": f"Could not score: {path}"}

    return {
        "path": row.get("path"),
        "face_count": row.get("face_count"),
        "eye_sharpness_min": row.get("eye_sharpness_min"),
        "eye_sharpness_max": row.get("eye_sharpness_max"),
        "whole_image_sharpness": row.get("whole_image_sharpness"),
        "fallback_used": row.get("fallback_used"),
        "camera": row.get("camera"),
        "taken_at": row.get("taken_at"),
        "phash": row.get("phash"),
    }


@mcp.tool()
def find_unsharp_subjects(folder: str, mode: str = "relative", percentile: int = 10) -> list[dict]:
    """
    Find photos in folder below sharpness threshold.

    Two modes:
    - "relative" (default): return bottom N percentile by eye_sharpness_min
      where eye is scoreable (fallback_used=False). Use percentile param (default 10).
    - "absolute": return all with eye_sharpness_min < 50.0 (tunable).

    Sorted by eye_sharpness_min ascending (sharpest culls first).

    folder: folder path to scan
    mode: "relative" or "absolute"
    percentile: used only in relative mode (default 10 = bottom decile)
    Returns: list of dicts, sorted by sharpness ascending
    """
    conn = _get_conn()
    photos = _cache.get_photos_in_folder(conn, folder)

    if mode == "relative":
        scoreable = [p for p in photos if p.get("fallback_used") is False and p.get("eye_sharpness_min")]
        if not scoreable:
            return []
        scoreable.sort(key=lambda p: float(p["eye_sharpness_min"]))
        cutoff_idx = max(0, len(scoreable) - (len(scoreable) * percentile // 100))
        return scoreable[:cutoff_idx]
    elif mode == "absolute":
        result = [p for p in photos if p.get("eye_sharpness_min") and float(p.get("eye_sharpness_min", 999)) < 50.0]
        result.sort(key=lambda p: float(p["eye_sharpness_min"]))
        return result
    else:
        return {"error": f"Unknown mode: {mode}"}


@mcp.tool()
def find_no_subject(folder: str) -> list[dict]:
    """
    Find photos where face detection failed (face_count == 0).

    Likely: environmental shots, back-of-head, heavy occlusion. These photos
    fall back to whole_image_sharpness scoring and require manual review.

    Sorted by whole_image_sharpness ascending (sharpest first).

    folder: folder path to scan
    Returns: list of dicts with face_count=0, sorted by whole_image_sharpness ascending
    """
    conn = _get_conn()
    photos = _cache.get_photos_in_folder(conn, folder)

    no_face = [p for p in photos if p.get("face_count") == 0]
    no_face.sort(key=lambda p: float(p.get("whole_image_sharpness", 999)))
    return no_face


@mcp.tool()
def get_metadata(path: str) -> dict:
    """
    Retrieve all cached metadata for one photo.

    Includes CV scores (face_count, eye_sharpness_min/max, whole_image_sharpness,
    fallback_used), EXIF data (iso, shutter, aperture, focal_length, camera,
    taken_at), and perceptual hash (phash).

    path: absolute path to JPG file
    Returns: dict with all cached columns, or error dict if not in cache
    """
    conn = _get_conn()
    row = _cache.get_photo(conn, path)

    if row is None:
        return {"error": f"Not in cache: {path}"}

    return row


if __name__ == "__main__":
    mcp.run()
