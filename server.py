#!/usr/bin/env python3
"""
triage-keepers MCP server — Evenings 3–7.
Tools: list_folders, get_thumbnail, index_folder, assess_subject_sharpness,
find_unsharp_subjects, find_no_subject, get_metadata, find_burst_groups,
rank_burst_group, get_pair, find_orphans.
"""

import io
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import os

from mcp.server.fastmcp import FastMCP, Image as MCPImage
from PIL import Image as PILImage
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import cache as _cache
from score_portraits import score_image, score_batch

mcp = FastMCP("triage-keepers")

_DB_PATH = Path("~/.triage-keepers/cache.db")
_MODEL_PATH = Path(__file__).parent / "face_landmarker.task"
_conn = None
_detector = None
_db_lock = Lock()  # Protects SQLite writes; SQLite is thread-safe but slow under contention


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
def get_thumbnail(path: str, size: int = 512, annotate_face: bool = True) -> MCPImage:
    """Return a JPEG thumbnail with optional face/eye bounding boxes drawn."""
    from PIL import ImageDraw
    img = PILImage.open(path)
    img = img.convert("RGB")
    orig_w, orig_h = img.size
    img.thumbnail((size, size), PILImage.LANCZOS)
    new_w, new_h = img.size
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    if annotate_face:
        conn = _get_conn()
        row = _cache.get_photo(conn, path)
        if row:
            draw = ImageDraw.Draw(img)
            if row.get("face_bboxes"):
                for x, y, w, h in json.loads(row["face_bboxes"]):
                    draw.rectangle(
                        [int(x * scale_x), int(y * scale_y),
                         int((x + w) * scale_x), int((y + h) * scale_y)],
                        outline="lime", width=2,
                    )
            if row.get("eye_bboxes"):
                for face_eyes in json.loads(row["eye_bboxes"]):
                    for x, y, w, h in face_eyes:
                        draw.rectangle(
                            [int(x * scale_x), int(y * scale_y),
                             int((x + w) * scale_x), int((y + h) * scale_y)],
                            outline="cyan", width=1,
                        )

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return MCPImage(data=buf.getvalue(), format="jpeg")


@mcp.tool()
def index_folder(path: str, recursive: bool = True, max_workers: int = None) -> dict:
    """
    Walk path for JPGs, score with CV using thread pool, batch-write cache.

    Uses ThreadPoolExecutor (one detector per worker thread) to parallelize CV
    scoring. Main thread filters files by mtime/size, batches them, submits
    batches to workers, and batch-writes results to SQLite with _db_lock to
    avoid serialization.

    Returns dict: {"total": count, "indexed": count, "skipped": count}
    """
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        return {"error": f"Not a directory: {path}"}

    glob = p.rglob("*") if recursive else p.glob("*")
    jpgs = [f for f in glob if f.suffix.lower() in {".jpg", ".jpeg"} and f.is_file()]

    conn = _get_conn()

    # Filter to photos needing reindex (main thread checks DB)
    to_score = []
    for f in jpgs:
        stat = f.stat()
        if _cache.needs_reindex(conn, str(f), stat.st_mtime, stat.st_size):
            to_score.append((f, stat.st_mtime, stat.st_size))

    skipped = len(jpgs) - len(to_score)
    indexed = 0

    if not to_score:
        return {"total": len(jpgs), "indexed": 0, "skipped": skipped}

    # Batch scoring with ThreadPoolExecutor
    # max_workers = max_workers or os.cpu_count() or 4
    max_workers = 2
    batch_size = max(1, len(to_score) // (max_workers * 2))
    batches = [to_score[i:i + batch_size] for i in range(0, len(to_score), batch_size)]

    all_rows = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for batch in batches:
            # Extract just the file paths for scoring
            batch_paths = [f for f, _, _ in batch]
            # Create detector per worker to avoid thread-safety issues
            detector = _get_detector()
            future = executor.submit(score_batch, batch_paths, detector)
            futures.append((future, batch))  # Keep batch for mtime/size pairing

        for future, batch in futures:
            batch_results = future.result()
            # Pair results with mtime/size
            for row, (_, mtime, size) in zip(batch_results, batch):
                if row is not None:
                    row["mtime"] = mtime
                    row["size"] = size
                    all_rows.append(row)

    # Batch-write to SQLite (main thread holds lock for all writes)
    with _db_lock:
        for row in all_rows:
            _cache.upsert_photo(conn, row)
            indexed += 1

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
        cutoff = max(1, len(scoreable) * percentile // 100)
        return scoreable[:cutoff]
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


@mcp.tool()
def find_burst_groups(folder: str, hamming: int = 5) -> list[list[str]]:
    """
    Find groups of near-duplicate photos (bursts) via pHash clustering.

    Groups photos by perceptual hash (Hamming distance <= hamming threshold).
    Each group represents a sequence of frames shot in rapid succession with
    nearly identical composition.

    Algorithm: greedy clustering — start with first ungrouped photo, find all
    others within hamming threshold (directly or transitively), form group,
    repeat until all grouped.

    folder: folder path to scan
    hamming: max Hamming distance to group (default 5; typical burst threshold)
    Returns: list of groups, each group is list of file paths, sorted by
             eye_sharpness_min descending (sharpest candidates first)
    """
    conn = _get_conn()
    photos = _cache.get_photos_in_folder(conn, folder)

    if not photos:
        return []

    groups = _cache.group_by_phash(photos, hamming)

    result = []
    for group in groups:
        if len(group) <= 1:
            result.append([group[0]["path"]])
        else:
            sorted_group = sorted(
                group,
                key=lambda p: float(p.get("eye_sharpness_min", 0)) if p.get("eye_sharpness_min") else 0,
                reverse=True
            )
            result.append([p["path"] for p in sorted_group])

    return result


@mcp.tool()
def rank_burst_group(file_paths: list[str]) -> list[dict]:
    """
    Rank photos within a burst group by eye sharpness and surface EXIF deltas.

    Takes a group of near-duplicates (from find_burst_groups) and ranks by
    eye_sharpness_min (highest first). Also computes ISO/shutter/aperture
    deltas to show exposure bracketing or camera adjustments.

    file_paths: list of absolute paths to JPGs in one burst group
    Returns: list of dicts, sorted by eye_sharpness_min descending, with
             fields: path, face_count, eye_sharpness_min, eye_sharpness_max,
             fallback_used, iso, shutter, aperture, exif_deltas
    """
    conn = _get_conn()
    photos = []
    for path in file_paths:
        row = _cache.get_photo(conn, path)
        if row:
            photos.append(row)

    if not photos:
        return []

    photos.sort(
        key=lambda p: float(p.get("eye_sharpness_min", 0)) if p.get("eye_sharpness_min") else 0,
        reverse=True
    )

    iso_vals = [p.get("iso") for p in photos if p.get("iso")]
    shutter_vals = [p.get("shutter") for p in photos if p.get("shutter")]
    aperture_vals = [p.get("aperture") for p in photos if p.get("aperture")]

    result = []
    for p in photos:
        result.append({
            "path": p.get("path"),
            "face_count": p.get("face_count"),
            "eye_sharpness_min": p.get("eye_sharpness_min"),
            "eye_sharpness_max": p.get("eye_sharpness_max"),
            "fallback_used": p.get("fallback_used"),
            "iso": p.get("iso"),
            "shutter": p.get("shutter"),
            "aperture": p.get("aperture"),
            "camera": p.get("camera"),
            "taken_at": p.get("taken_at"),
            "exif_deltas": {
                "iso_range": f"{min(iso_vals)}-{max(iso_vals)}" if iso_vals else "N/A",
                "shutter_range": f"{min(shutter_vals)}-{max(shutter_vals)}" if shutter_vals else "N/A",
                "aperture_range": f"{min(aperture_vals)}-{max(aperture_vals)}" if aperture_vals else "N/A",
            }
        })

    return result


@mcp.tool()
def get_pair(basename: str, folder: str) -> dict:
    """
    Find RAW + JPG pairing for a photo.

    Given a JPG basename (e.g. "IMG_7102.JPG"), look for a sibling RAW file
    with the same basename but .CR2, .NEF, .ARW extension (Canon, Nikon, Sony).
    Return pairing status and file paths.

    basename: filename with extension (e.g. "IMG_7102.JPG")
    folder: folder to search (should contain both JPG and RAW files)
    Returns: dict with jpg, raw, status (one of "paired", "jpg_only", "raw_only"),
             jpg_size, raw_size
    """
    folder_path = Path(folder).resolve()
    name_stem = Path(basename).stem

    jpg_path = None
    raw_path = None

    for f in folder_path.iterdir():
        if f.stem.upper() == name_stem.upper():
            if f.suffix.upper() in {".JPG", ".JPEG"}:
                jpg_path = f
            elif f.suffix.upper() in {".CR2", ".NEF", ".ARW", ".RW2", ".DNG"}:
                raw_path = f

    jpg_size = jpg_path.stat().st_size if jpg_path else None
    raw_size = raw_path.stat().st_size if raw_path else None

    if jpg_path and raw_path:
        status = "paired"
    elif jpg_path:
        status = "jpg_only"
    elif raw_path:
        status = "raw_only"
    else:
        status = "not_found"

    return {
        "basename": basename,
        "jpg": str(jpg_path) if jpg_path else None,
        "raw": str(raw_path) if raw_path else None,
        "status": status,
        "jpg_size": jpg_size,
        "raw_size": raw_size,
    }


@mcp.tool()
def find_orphans(folder: str) -> dict:
    """
    Find unpaired RAW and JPG files in folder.

    RAWs without JPGs: user may have deleted JPG after processing.
    JPGs without RAWs: shot in JPG-only mode, or RAW deleted/moved.

    folder: folder path to scan
    Returns: dict with raw_orphans (list of RAW paths) and jpg_orphans
             (list of JPG paths with no RAW sibling)
    """
    folder_path = Path(folder).resolve()

    raw_extensions = {".CR2", ".NEF", ".ARW", ".RW2", ".DNG"}
    jpg_extensions = {".JPG", ".JPEG"}

    stems_with_jpg = set()
    stems_with_raw = set()

    for f in folder_path.iterdir():
        if f.suffix.upper() in jpg_extensions:
            stems_with_jpg.add(f.stem.upper())
        elif f.suffix.upper() in raw_extensions:
            stems_with_raw.add(f.stem.upper())

    raw_orphans = []
    jpg_orphans = []

    for f in folder_path.iterdir():
        stem = f.stem.upper()
        if f.suffix.upper() in raw_extensions and stem not in stems_with_jpg:
            raw_orphans.append(str(f))
        elif f.suffix.upper() in jpg_extensions and stem not in stems_with_raw:
            jpg_orphans.append(str(f))

    return {
        "raw_orphans": sorted(raw_orphans),
        "jpg_orphans": sorted(jpg_orphans),
        "raw_count": len(raw_orphans),
        "jpg_count": len(jpg_orphans),
    }


@mcp.tool()
def summarize_folder(folder: str) -> dict:
    """
    Aggregate stats for all indexed photos in a folder.

    Returns counts, sharpness distribution (min/max/mean/median for
    eye-scored photos), and burst group count. Useful as a first call
    before running sharpness or burst flows.

    folder: folder path to scan
    Returns: dict with total, face_detected, fallback_count, no_face_count,
             sharpness_min, sharpness_max, sharpness_mean, sharpness_median,
             burst_group_count (groups of 2+)
    """
    import statistics
    conn = _get_conn()
    photos = _cache.get_photos_in_folder(conn, folder)

    if not photos:
        return {"total": 0, "error": "No indexed photos in folder. Run index_folder first."}

    total = len(photos)
    face_detected = sum(1 for p in photos if p.get("face_count", 0) and int(p["face_count"]) > 0)
    no_face = sum(1 for p in photos if not p.get("face_count") or int(p.get("face_count", 0)) == 0)
    fallback_count = sum(1 for p in photos if p.get("fallback_used"))

    eye_scores = [
        float(p["eye_sharpness_min"])
        for p in photos
        if p.get("eye_sharpness_min") and not p.get("fallback_used")
    ]

    sharpness_stats = {}
    if eye_scores:
        sharpness_stats = {
            "sharpness_min":    round(min(eye_scores), 2),
            "sharpness_max":    round(max(eye_scores), 2),
            "sharpness_mean":   round(statistics.mean(eye_scores), 2),
            "sharpness_median": round(statistics.median(eye_scores), 2),
        }

    groups = _cache.group_by_phash(photos)
    burst_group_count = sum(1 for g in groups if len(g) >= 2)

    return {
        "total":             total,
        "face_detected":     face_detected,
        "no_face_count":     no_face,
        "fallback_count":    fallback_count,
        "burst_group_count": burst_group_count,
        **sharpness_stats,
    }


if __name__ == "__main__":
    mcp.run()
