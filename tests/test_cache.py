import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from cache import init_db, needs_reindex, upsert_photo


@pytest.fixture
def conn(tmp_path):
    return init_db(tmp_path / "test.db")


def test_init_creates_table(conn):
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert ("photos",) in tables


def test_init_creates_index(conn):
    indexes = conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
    assert any("idx_photos_dir" in row[0] for row in indexes)


def test_init_idempotent(tmp_path):
    db = tmp_path / "test.db"
    init_db(db)
    init_db(db)  # should not raise


def test_upsert_inserts_row(conn):
    upsert_photo(conn, {"path": "/a.jpg", "mtime": 1.0, "size": 100})
    row = conn.execute("SELECT path, mtime, size FROM photos WHERE path='/a.jpg'").fetchone()
    assert row == ("/a.jpg", 1.0, 100)


def test_upsert_replaces_on_collision(conn):
    upsert_photo(conn, {"path": "/a.jpg", "mtime": 1.0, "size": 100})
    upsert_photo(conn, {"path": "/a.jpg", "mtime": 2.0, "size": 200})
    rows = conn.execute("SELECT count(*) FROM photos WHERE path='/a.jpg'").fetchone()
    assert rows[0] == 1
    row = conn.execute("SELECT mtime, size FROM photos WHERE path='/a.jpg'").fetchone()
    assert row == (2.0, 200)


def test_upsert_sets_indexed_at(conn):
    upsert_photo(conn, {"path": "/a.jpg", "mtime": 1.0, "size": 100})
    val = conn.execute("SELECT indexed_at FROM photos WHERE path='/a.jpg'").fetchone()[0]
    assert val is not None


def test_needs_reindex_new_path(conn):
    assert needs_reindex(conn, "/unknown.jpg", 1.0, 100) is True


def test_needs_reindex_same(conn):
    upsert_photo(conn, {"path": "/a.jpg", "mtime": 1.0, "size": 100})
    assert needs_reindex(conn, "/a.jpg", 1.0, 100) is False


def test_needs_reindex_changed_mtime(conn):
    upsert_photo(conn, {"path": "/a.jpg", "mtime": 1.0, "size": 100})
    assert needs_reindex(conn, "/a.jpg", 9.0, 100) is True


def test_needs_reindex_changed_size(conn):
    upsert_photo(conn, {"path": "/a.jpg", "mtime": 1.0, "size": 100})
    assert needs_reindex(conn, "/a.jpg", 1.0, 999) is True


def test_group_by_phash_with_time_window(conn):
    """
    group_by_phash with time_window_seconds should split groups taken far apart.

    Creates 3 photos with same pHash but different taken_at times:
    - photo 0: 2026-05-25 10:00:00
    - photo 1: 2026-05-25 10:00:30 (30s later, within 60s window)
    - photo 2: 2026-05-25 10:05:00 (5min later, outside 60s window from photo 1)

    Without time window: all 3 group together (transitive: 0-1-2).
    With time_window=60: photos 0-1 group, photo 2 alone.
    """
    from cache import group_by_phash

    photos = [
        {
            "path": "/photo_0.jpg",
            "phash": "1234567890abcdef",  # Same pHash for all
            "taken_at": "2026:05:25 10:00:00"
        },
        {
            "path": "/photo_1.jpg",
            "phash": "1234567890abcdef",
            "taken_at": "2026:05:25 10:00:30"
        },
        {
            "path": "/photo_2.jpg",
            "phash": "1234567890abcdef",
            "taken_at": "2026:05:25 10:05:00"
        }
    ]

    # Without time window: all group together
    groups_no_window = group_by_phash(photos, hamming_threshold=10)
    assert len(groups_no_window) == 1
    assert len(groups_no_window[0]) == 3

    # With time window: split into two groups
    groups_with_window = group_by_phash(photos, hamming_threshold=10, time_window_seconds=60)
    assert len(groups_with_window) == 2
    # One group has 2 photos (0-1), other has 1 (2)
    group_sizes = sorted([len(g) for g in groups_with_window])
    assert group_sizes == [1, 2]
