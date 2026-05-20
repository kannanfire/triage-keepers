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
