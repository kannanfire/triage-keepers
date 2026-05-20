#!/usr/bin/env python3
"""
triage-keepers MCP server — Evenings 3–5.
Tools: list_folders, get_thumbnail, index_folder.
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


if __name__ == "__main__":
    mcp.run()
