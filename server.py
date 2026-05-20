#!/usr/bin/env python3
"""
triage-keepers MCP server — Evenings 3 + 4.
Tools: list_folders, get_thumbnail, index_folder.
"""

import io
from pathlib import Path

from mcp.server.fastmcp import FastMCP, Image as MCPImage
from PIL import Image as PILImage

import cache as _cache

mcp = FastMCP("triage-keepers")

_DB_PATH = Path("~/.triage-keepers/cache.db")
_conn = None


def _get_conn():
    global _conn
    if _conn is None:
        _conn = _cache.init_db(_DB_PATH)
    return _conn


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
    """Walk path for JPGs, upsert skeleton cache rows. Returns counts."""
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        return {"error": f"Not a directory: {path}"}

    glob = p.rglob("*") if recursive else p.glob("*")
    jpgs = [f for f in glob if f.suffix.lower() in {".jpg", ".jpeg"} and f.is_file()]

    conn = _get_conn()
    indexed = 0
    skipped = 0

    for f in jpgs:
        stat = f.stat()
        if _cache.needs_reindex(conn, str(f), stat.st_mtime, stat.st_size):
            _cache.upsert_photo(conn, {
                "path": str(f),
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            })
            indexed += 1
        else:
            skipped += 1

    return {"total": len(jpgs), "indexed": indexed, "skipped": skipped}


if __name__ == "__main__":
    mcp.run()
