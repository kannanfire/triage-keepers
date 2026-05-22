# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

triage-keepers is an MCP (Model Context Protocol) server for portrait photo culling. It runs locally (no cloud, no API keys) and integrates with Claude Desktop. Two co-equal workflows:

1. **Sharpness flow** — score subject eye-region focus, ignoring intentional bokeh
2. **Burst flow** — group near-duplicate frames via perceptual hashing, rank within groups by sharpness

Central design: instead of whole-image sharpness (which mistakes bokeh for blur), score the eye region using MediaPipe face landmarks + Laplacian variance.

## Setup & Dependencies

**Install:** Python ≥3.11 managed by `uv`:
```bash
uv sync
```

**Model file:** MediaPipe requires a ~26MB task file. Download once:
```bash
curl -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

The server looks for this file in the repo root. `score_portraits.py` can auto-fetch it on first run if missing (see `MODEL_URL` constant).

## Running & Testing

**Run tests:**
```bash
uv run pytest tests/ -v
uv run pytest tests/test_cache.py -v
uv run pytest tests/test_server_tools.py::test_index_folder_first_run -v  # Single test
```

**Manual scoring (offline):**
```bash
uv run python score_portraits.py <folder> [output.csv]
# Walks folder, scores all JPGs, writes path + eye sharpness + EXIF to CSV
```

**MCP server (Claude Desktop integration):**
```bash
uv run mcp run python server.py
# Runs FastMCP server listening on stdio. Wire into claude_desktop_config.json.
```

**Inspect SQLite cache:**
```bash
sqlite3 ~/.triage-keepers/cache.db "SELECT path, face_count, eye_sharpness_min FROM photos LIMIT 10;"
```

## Key Files & Architecture

### Main Modules

- **`server.py`** — FastMCP server exposing 10 tools (index_folder, assess_subject_sharpness, find_burst_groups, get_thumbnail, etc.). Single module-level detector + SQLite connection for efficiency.
- **`score_portraits.py`** — Standalone CV pipeline: loads JPG → detects faces/eyes → crops eye region → computes Laplacian variance → extracts EXIF. Also handles fallback to whole-image sharpness when face detection fails.
- **`cache.py`** — SQLite schema + upsert logic. Cache key is `(absolute_path, mtime, size)`. Detects file changes and re-scores only when needed.

### CV Pipeline (the core claim)

```
JPG → MediaPipe Face Landmarker (478 landmarks per face)
   → extract eye-contour indices → compute eye-region bbox (with padding)
   → crop to eye region → resize to 128×128
   → Laplacian variance → sharpness score
   → pHash (whole image, for burst grouping)
   → cache result
```

**Face detection:** MediaPipe's `FaceLandmarker` (local, free, ~0.5s per JPG on CPU). Handles multi-face, profile view, occlusion. Falls back to whole-image Laplacian if eyes not detected.

**Sharpness metric:** Laplacian variance. High-contrast edges (sharp) = high variance; smooth/blur = low variance. Normalized to 128×128 for consistency across camera resolutions.

**Eye region selection:** Canonical face-mesh indices (LEFT_EYE, RIGHT_EYE in constants). Bounding box = min/max x,y across indices, padded 20%, clamped to image bounds.

### Test Structure

- **`test_score_portraits.py`** — Unit tests for CV functions (Laplacian, eye bbox, fallback logic). Uses synthetic sharp/blurry fixture images.
- **`test_server_tools.py`** — Integration tests for MCP server tools. Each test gets isolated temp DB (via `isolated_db` fixture) to avoid state bleed. Tests `index_folder`, `list_folders`, `get_thumbnail`, `find_burst_groups`, etc.
- **`test_cache.py`** — Cache upsert, cache-hit detection, DB schema.
- **`fixtures/`** — Synthetic JPEG images (sharp.jpg, blurry.jpg) for testing; no dependency on real photos.

### Important Constants

**`score_portraits.py`:**
- `NORM_EYE_SIZE = 128` — eye crops resized to this before Laplacian
- `NORM_WHOLE_SIZE = 512` — whole-image fallback normalization
- `MIN_EYE_WIDTH = 20` — skip eye bboxes narrower than this (pixel)
- `EYE_PAD = 0.20` — 20% padding around raw eye-contour bbox
- `VISIBILITY_MIN = 0.5` — landmark visibility threshold

**`server.py`:**
- `_DB_PATH = ~/.triage-keepers/cache.db` — SQLite lives here
- `_MODEL_PATH = face_landmarker.task` — model file in repo root
- ThreadPoolExecutor for CV parallelization; each worker gets own detector to avoid GIL contention

## Architecture Decisions

**SQLite cache:** Caches CV results (face_count, eye_sharpness_min/max, pHash, EXIF) by file path + mtime + size. Two-phase design: **index** (walk folder, score, cache — slow) and **query** (read cache — fast). Re-scores only on file change.

**Thread pool in index_folder:** Main thread filters files by mtime/size, batches them, submits to workers. Each worker runs MediaPipe independently (no shared detector). Main thread writes cache with lock protection (SQLite is thread-safe but slow under contention).

**Module-level singletons:** `server.py` caches one DB connection and one MediaPipe detector for lifetime of process (lazy init). Tests reset these between runs with `_reset_server_conn()`.

**Fallback to whole-image scoring:** If face detection fails (no face, sunglasses, heavy occlusion), use Laplacian on normalized whole image. Marked with `fallback_used=true` so Claude can decide weight.

## Spec & Design Goals

See `triage-keepers-spec_4.md` for full design:
- Two workflows (sharpness + burst)
- Server is read-only — never deletes, user culls manually
- Non-goals: cloud, automated deletion, video, cross-folder burst, editing, aesthetic judgment

## Confidence Levels

From spec:
- **Face detection framing for portraiture:** High
- **Eye-region Laplacian tracking human judgment:** Medium (awaiting user validation)
- **All tools (index, assess, list, find_burst, rank, etc.):** High-Medium depending on feature
- **pHash burst grouping:** High
- **Thumbnail rendering with face boxes:** High (load-bearing for Claude visual confirmation)
