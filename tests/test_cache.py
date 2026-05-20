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
