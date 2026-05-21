import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
  path TEXT PRIMARY KEY,
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
  camera TEXT, taken_at TEXT, indexed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_photos_dir ON photos(path);
"""

_COLUMNS = [
    "path", "mtime", "size", "face_count",
    "eye_sharpness_min", "eye_sharpness_max", "whole_image_sharpness", "fallback_used",
    "phash", "face_bboxes", "eye_bboxes",
    "iso", "shutter", "aperture", "focal_length",
    "camera", "taken_at", "indexed_at",
]


def init_db(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def upsert_photo(conn: sqlite3.Connection, row: dict) -> None:
    full = {col: row.get(col) for col in _COLUMNS}
    if "indexed_at" not in row:
        full["indexed_at"] = datetime.now(timezone.utc).isoformat()
    placeholders = ", ".join(["?"] * len(_COLUMNS))
    cols = ", ".join(_COLUMNS)
    conn.execute(
        f"INSERT OR REPLACE INTO photos ({cols}) VALUES ({placeholders})",
        [full[c] for c in _COLUMNS],
    )
    conn.commit()


def needs_reindex(conn: sqlite3.Connection, path: str, mtime: float, size: int) -> bool:
    row = conn.execute(
        "SELECT mtime, size FROM photos WHERE path = ?", (path,)
    ).fetchone()
    if row is None:
        return True
    return row[0] != mtime or row[1] != size


def get_photo(conn: sqlite3.Connection, path: str) -> dict | None:
    """
    Retrieve one photo record by absolute path.

    path: absolute file path (as stored in cache)
    Returns: dict with all columns, or None if not in cache
    """
    row = conn.execute(
        "SELECT * FROM photos WHERE path = ?", (path,)
    ).fetchone()
    if row is None:
        return None
    cols = _COLUMNS
    return dict(zip(cols, row))


def get_photos_in_folder(conn: sqlite3.Connection, folder: str) -> list[dict]:
    """
    Retrieve all photos cached under a folder path (recursive).

    folder: folder path (will match paths starting with this prefix)
    Returns: list of dicts, one per photo
    """
    folder_path = Path(folder).resolve()
    prefix = str(folder_path) + "/"
    rows = conn.execute(
        "SELECT * FROM photos WHERE path LIKE ? ORDER BY path",
        (f"{prefix}%",)
    ).fetchall()
    cols = _COLUMNS
    return [dict(zip(cols, row)) for row in rows]


def get_all_photos(conn: sqlite3.Connection) -> list[dict]:
    """
    Retrieve all cached photos (unfiltered).

    Returns: list of dicts, one per photo
    """
    rows = conn.execute("SELECT * FROM photos ORDER BY path").fetchall()
    cols = _COLUMNS
    return [dict(zip(cols, row)) for row in rows]


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


def group_by_phash(photos: list[dict], hamming_threshold: int = 5) -> list[list[dict]]:
    """
    Group photos by pHash similarity (Hamming distance <= threshold).

    Greedy clustering: start with first ungrouped photo, find all photos
    within hamming_threshold of it (directly or transitively), form group.

    photos: list of photo dicts from cache
    hamming_threshold: max Hamming distance to group (default 5)
    Returns: list of groups, each group is list of photo dicts
    """
    if not photos:
        return []

    phash_photos = [p for p in photos if p.get("phash")]
    if not phash_photos:
        return []

    groups = []
    ungrouped = set(range(len(phash_photos)))

    while ungrouped:
        seed_idx = min(ungrouped)
        group_indices = {seed_idx}
        queue = [seed_idx]

        while queue:
            curr_idx = queue.pop(0)
            seed_phash = phash_photos[curr_idx]["phash"]

            for other_idx in list(ungrouped):
                if other_idx == curr_idx:
                    continue
                other_phash = phash_photos[other_idx]["phash"]
                if hamming_distance(seed_phash, other_phash) <= hamming_threshold:
                    group_indices.add(other_idx)
                    queue.append(other_idx)

        ungrouped -= group_indices
        groups.append([phash_photos[i] for i in sorted(group_indices)])

    return groups
