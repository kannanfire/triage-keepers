#!/usr/bin/env python3
"""
triage-keepers MCP server — Evening 3 skeleton.
Two tools: list_folders, get_thumbnail.
No CV, no cache yet.
"""

import io
from pathlib import Path

from mcp.server.fastmcp import FastMCP, Image as MCPImage
from PIL import Image as PILImage

mcp = FastMCP("triage-keepers")


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


if __name__ == "__main__":
    mcp.run()
