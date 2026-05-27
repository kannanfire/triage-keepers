import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
  library_root TEXT NOT NULL,
  rel_path TEXT NOT NULL,
  mtime REAL, size INTEGER,
  face_count INTEGER,
  eye_sharpness_min REAL,
  eye_sharpness_max REAL,
  whole_image_sharpness REAL,
  fallback_used INTEGER,
  phash TEXT,
  face_bboxes TEXT,
  eye_bboxes TEXT,
  iso INTEGER, shutter TEXT, aperture REAL, focal_length REAL,
  camera TEXT, taken_at TEXT, indexed_at TEXT,
  PRIMARY KEY (library_root, rel_path)
);
CREATE INDEX IF NOT EXISTS idx_photos_dir ON photos(library_root, rel_path);
"""

_COLUMNS = [
    "library_root", "rel_path", "mtime", "size", "face_count",
    "eye_sharpness_min", "eye_sharpness_max", "whole_image_sharpness", "fallback_used",
    "phash", "face_bboxes", "eye_bboxes",
    "iso", "shutter", "aperture", "focal_length",
    "camera", "taken_at", "indexed_at",
]

_FOLDER_COLS = [
    "library_root", "rel_path",
    "face_count", "eye_sharpness_min", "eye_sharpness_max",
    "whole_image_sharpness", "fallback_used",
    "phash", "iso", "shutter", "aperture", "focal_length", "camera", "taken_at",
]


def init_db(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def upsert_photo(conn: sqlite3.Connection, row: dict, library_root: str | Path = None) -> None:
    """
    Upsert a photo record. Automatically computes rel_path from library_root.

    row: dict with photo data. If 'path' is absolute and library_root provided,
         'rel_path' is computed automatically. Otherwise, 'rel_path' must be in row.
    library_root: optional absolute path to compute relative path. If not provided,
                  uses row.get('library_root') or infers from row.get('path').
    """
    try:
        full = {col: row.get(col) for col in _COLUMNS}

        # Determine library_root if not provided
        if library_root is None:
            library_root = row.get("library_root")
        if library_root is None and row.get("path"):
            # Infer from existing cached rows or use first path component
            library_root = row.get("path")  # Caller must set explicitly if needed

        library_root = str(Path(library_root).resolve()) if library_root else None

        # Compute rel_path if not already in row
        if "rel_path" not in row and row.get("path") and library_root:
            try:
                abs_path = Path(row["path"]).resolve()
                lib_path = Path(library_root).resolve()
                rel_path = abs_path.relative_to(lib_path)
                full["rel_path"] = str(rel_path)
            except ValueError as e:
                print(f"upsert_photo: failed to compute relative path for {row.get('path')} under {library_root}: {e}", file=sys.stderr)
                full["rel_path"] = row.get("path")
            except TypeError as e:
                print(f"upsert_photo: type error computing relative path: {e}", file=sys.stderr)
                full["rel_path"] = row.get("path")

        full["library_root"] = library_root or ""

        if "indexed_at" not in row:
            full["indexed_at"] = datetime.now(timezone.utc).isoformat()

        placeholders = ", ".join(["?"] * len(_COLUMNS))
        cols = ", ".join(_COLUMNS)
        conn.execute(
            f"INSERT OR REPLACE INTO photos ({cols}) VALUES ({placeholders})",
            [full[c] for c in _COLUMNS],
        )
        conn.commit()
    except Exception as e:
        print(f"upsert_photo error for path {row.get('path')}: {e}", file=sys.stderr)


def needs_reindex(conn: sqlite3.Connection, path: str, mtime: float, size: int, library_root: str | Path = None) -> bool:
    """
    Check if a photo needs reindexing (missing from cache or file changed).

    path: absolute file path
    library_root: optional folder root. If not provided, infers from path directory.
    Returns: True if needs reindexing, False if cached and unchanged
    """
    try:
        if library_root is None:
            library_root = str(Path(path).parent)

        library_root = str(Path(library_root).resolve())
        abs_path = Path(path).resolve()
        lib_path = Path(library_root).resolve()

        try:
            rel_path = str(abs_path.relative_to(lib_path))
        except ValueError as e:
            print(f"needs_reindex: path {path} not under library_root {library_root}: {e}", file=sys.stderr)
            return True

        row = conn.execute(
            "SELECT mtime, size FROM photos WHERE library_root = ? AND rel_path = ?",
            (library_root, rel_path)
        ).fetchone()
        if row is None:
            return True
        return row[0] != mtime or row[1] != size
    except Exception as e:
        print(f"needs_reindex error for {path}: {e}", file=sys.stderr)
        return True


def get_photo(conn: sqlite3.Connection, path: str, library_root: str | Path = None) -> dict | None:
    """
    Retrieve one photo record by absolute path.

    path: absolute file path
    library_root: optional folder root. If not provided, infers from path directory.
    Returns: dict with all columns, or None if not in cache
    """
    try:
        if library_root is None:
            library_root = str(Path(path).parent)

        library_root = str(Path(library_root).resolve())
        abs_path = Path(path).resolve()
        lib_path = Path(library_root).resolve()

        try:
            rel_path = str(abs_path.relative_to(lib_path))
        except ValueError as e:
            print(f"get_photo: path {path} not under library_root {library_root}: {e}", file=sys.stderr)
            return None

        row = conn.execute(
            "SELECT * FROM photos WHERE library_root = ? AND rel_path = ?",
            (library_root, rel_path)
        ).fetchone()
        if row is None:
            return None
        cols = _COLUMNS
        return dict(zip(cols, row))
    except Exception as e:
        print(f"get_photo error for {path}: {e}", file=sys.stderr)
        return None


def get_photo_by_abspath(conn: sqlite3.Connection, path: str) -> dict | None:
    """
    Retrieve one photo record by absolute path (full-path match).

    Queries using reconstructed full path (library_root || '/' || rel_path).
    Use this when library_root is unknown but the absolute path is available.

    path: absolute file path
    Returns: dict with all columns, or None if not in cache
    """
    try:
        abs_path = str(Path(path).resolve())
        row = conn.execute(
            "SELECT * FROM photos WHERE library_root || '/' || rel_path = ?",
            (abs_path,)
        ).fetchone()
        if row is None:
            return None
        cols = _COLUMNS
        return dict(zip(cols, row))
    except Exception as e:
        print(f"get_photo_by_abspath error for {path}: {e}", file=sys.stderr)
        return None


def get_photos_in_folder(conn: sqlite3.Connection, folder: str, library_root: str | Path = None) -> list[dict]:
    """
    Retrieve all photos cached under a folder path (recursive).

    Supports both exact-match (indexed at this folder) and parent-folder queries
    (for summarize_folder on parent of indexed folders).

    folder: folder path to query
    library_root: deprecated, ignored. Kept for backward compatibility.
    Returns: list of dicts, one per photo under folder
    """
    try:
        folder_path = Path(folder).resolve()
        folder_str = str(folder_path)

        # Query all library_roots that match the folder exactly or are subdirectories
        cols_str = ", ".join(_FOLDER_COLS)
        rows = conn.execute(
            f"SELECT {cols_str} FROM photos WHERE library_root = ? OR library_root LIKE ? || '/%' ORDER BY library_root, rel_path",
            (folder_str, folder_str)
        ).fetchall()

        result = []
        for row in rows:
            try:
                photo_dict = dict(zip(_FOLDER_COLS, row))
                # Reconstruct full path and check if it's under folder (secondary guard)
                full_path = Path(photo_dict["library_root"]) / photo_dict["rel_path"]
                if str(full_path).startswith(str(folder_path)):
                    result.append(photo_dict)
            except Exception as e:
                print(f"get_photos_in_folder: error processing row: {e}", file=sys.stderr)
                continue

        return result
    except Exception as e:
        print(f"get_photos_in_folder error for folder {folder}: {e}", file=sys.stderr)
        return []


# DEPRECATED: get_all_photos had ORDER BY path which doesn't exist in P11 schema.
# def get_all_photos(conn: sqlite3.Connection) -> list[dict]:
#     """
#     Retrieve all cached photos (unfiltered).
#
#     Returns: list of dicts, one per photo
#     """
#     rows = conn.execute("SELECT * FROM photos ORDER BY path").fetchall()
#     cols = _COLUMNS
#     return [dict(zip(cols, row)) for row in rows]


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Compute Hamming distance between two hex pHash strings.

    Hamming distance = count of bit positions that differ.
    Two identical images: distance 0.
    Near-duplicates (same burst): distance 1–5.
    Different images: distance 30+.

    hash1, hash2: hex strings (e.g. "a1b2c3d4e5f6a7b8")
    Returns: int Hamming distance (0–64 for 64-bit pHash)
    """
    try:
        val1 = int(hash1, 16)
        val2 = int(hash2, 16)
        xor = val1 ^ val2
        return bin(xor).count('1')
    except (ValueError, TypeError):
        return 999


def group_by_phash(photos: list[dict], hamming_threshold: int = 5, time_window_seconds: int = None) -> list[list[dict]]:
    """
    Group photos by pHash similarity (Hamming distance <= threshold).

    Greedy clustering: start with first ungrouped photo, find all photos
    within hamming_threshold of it (directly or transitively), form group.
    Optionally constrain by taken_at timestamp.

    photos: list of photo dicts from cache
    hamming_threshold: max Hamming distance to group (default 5)
    time_window_seconds: optional max seconds between taken_at timestamps.
                         Photos without taken_at fall back to pHash-only grouping.
    Returns: list of groups, each group is list of photo dicts
    """
    if not photos:
        return []

    phash_photos = [p for p in photos if p.get("phash")]
    if not phash_photos:
        return []

    def parse_timestamp(taken_at_str):
        """Parse EXIF timestamp 'YYYY:MM:DD HH:MM:SS' to seconds since epoch."""
        if not taken_at_str:
            return None
        try:
            from datetime import datetime
            dt = datetime.strptime(taken_at_str, "%Y:%m:%d %H:%M:%S")
            return dt.timestamp()
        except (ValueError, TypeError):
            return None

    def within_time_window(ts1, ts2, window_seconds):
        """Check if two timestamps are within window_seconds of each other."""
        if ts1 is None or ts2 is None:
            return True  # Missing timestamp falls back to pHash-only
        return abs(ts1 - ts2) <= window_seconds

    groups = []
    ungrouped = set(range(len(phash_photos)))

    while ungrouped:
        seed_idx = min(ungrouped)
        group_indices = {seed_idx}
        queue = [seed_idx]
        seed_timestamp = parse_timestamp(phash_photos[seed_idx].get("taken_at")) if time_window_seconds else None

        while queue:
            curr_idx = queue.pop(0)
            curr_phash = phash_photos[curr_idx]["phash"]
            curr_timestamp = parse_timestamp(phash_photos[curr_idx].get("taken_at")) if time_window_seconds else None

            for other_idx in list(ungrouped):
                if other_idx in group_indices:
                    continue
                other_phash = phash_photos[other_idx]["phash"]
                other_timestamp = parse_timestamp(phash_photos[other_idx].get("taken_at")) if time_window_seconds else None

                hamming_match = hamming_distance(curr_phash, other_phash) <= hamming_threshold
                time_match = True
                if time_window_seconds is not None:
                    time_match = within_time_window(curr_timestamp, other_timestamp, time_window_seconds)

                if hamming_match and time_match:
                    group_indices.add(other_idx)
                    queue.append(other_idx)

        ungrouped -= group_indices
        groups.append([phash_photos[i] for i in sorted(group_indices)])

    return groups
