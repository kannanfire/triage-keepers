# triage-keepers

An MCP server for portrait photo culling. Two co-equal workflows: subject-sharpness assessment and burst-group ranking. Local-only; consumed by Claude Desktop. No API key required.

---

## Goals

1. **Sharpness flow** — score whether the subject's eye region is in focus, ignoring intentionally blurred bokeh.
2. **Burst flow** — group near-duplicate frames, rank within each group, surface the best candidate for Claude to confirm visually.
3. Server is read-only. Never deletes. User culls manually.

**Non-goals:** cloud deployment, automated deletion, video files, cross-folder burst detection, photo editing, aesthetic judgment beyond what's measurable.

---

## Core design decision

Generic photo-culling tools fail on portraits because whole-image sharpness scoring mistakes intentional bokeh for blur. **The central choice: score the eye region, not the image.** Both flows share one CV pipeline (detect face → locate eyes → score that region) and use it differently.

Confidence the framing is correct for portraiture: **High**. Confidence the eye-region-Laplacian pipeline tracks human judgment well enough to be useful: **Medium until validated** (see Risks).

---

## Architecture

```
Claude Desktop ──MCP/stdio──► triage-keepers server
                                    │
                                    ├─► SQLite cache
                                    └─► Filesystem (read-only)
```

Two-phase: **index** (walk library, score every JPG, cache results — slow, one-time per file) and **query** (agent reads cache — fast). Cache key: `(absolute_path, mtime, size)`. Re-score only on file change.

---

## Tools

| Tool | Purpose | Flow | Confidence |
|---|---|---|---|
| `index_folder(path, recursive=true)` | Walk and index. Returns progress. | Both | High |
| `list_folders(root)` | Show hierarchy. | Both | High |
| `assess_subject_sharpness(path)` | Per-face eye-region sharpness, face count, fallback flag. | Sharpness | Medium-High |
| `find_unsharp_subjects(folder, mode="relative")` | Photos below threshold (bottom decile by default). | Sharpness | Medium |
| `find_no_subject(folder)` | Photos where face detection failed. | Sharpness | High |
| `find_burst_groups(folder, hamming=5)` | Group near-duplicates via pHash. | Burst | High |
| `rank_burst_group(file_paths)` | Rank within a group by eye-sharpness, with EXIF deltas. | Burst | High |
| `get_pair(basename, folder)` | RAW + JPG pairing info. | Both | High |
| `find_orphans(folder)` | RAWs without JPGs or vice versa. | Both | High |
| `get_thumbnail(path, size=512, annotate_face=true)` | Base64 JPG with face/eye bounding boxes drawn. Load-bearing tool — Claude's visual confirmation depends on it. | Both | High |
| `get_metadata(path)` | EXIF + file stats. | Both | High |
| `summarize_folder(folder)` | Aggregate stats. | Both | High |

All read-only. No `delete_*` tool. Deliberate.

---

## CV pipeline

```
JPG → MediaPipe Face Landmarker → eye landmarks → eye-region bbox
    → crop → Laplacian variance → sharpness score
    → pHash (whole image, for burst grouping)
    → cache
```

### Face detection: MediaPipe Face Landmarker

478 facial landmarks per face including eye-specific landmarks. Multi-face capable. Free, local, no cloud. Requires the `face_landmarker.task` model file (~26MB) downloaded separately from `storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task` — not bundled with the pip package. Eye-region bbox computed from canonical face-mesh eye-contour indices:

```
LEFT_EYE  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
```

Bbox = min/max x,y across the index set, multiplied by image dimensions (MediaPipe returns normalized [0.0, 1.0] coordinates).

**Handling:**
- Multi-face: store `min(eye_sharpness)` and `max(eye_sharpness)` per photo. Agent decides which matters.
- Profile: MediaPipe returns single eye landmark set. Works.
- Heavy occlusion (sunglasses, hair, hands): no eye landmarks → fallback to whole-face Laplacian, `fallback_used=true`.
- No face detected: fallback to whole-image score, surfaced via `find_no_subject` for review.

Detection-rate confidence: **High** for frontal/three-quarter, **Medium-High** for profile, **Medium** for heavy occlusion.

Source: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

### Eye-region sharpness: Laplacian variance on crop

```python
import cv2
NORM_SIZE = 128  # eye crops resized to NORM_SIZE × NORM_SIZE before Laplacian

def eye_sharpness(image, face_landmarks, eye_indices):
    img_h, img_w = image.shape[:2]
    # MediaPipe landmarks are normalized [0.0, 1.0]; convert to pixel coords
    xs = [int(face_landmarks[i].x * img_w) for i in eye_indices]
    ys = [int(face_landmarks[i].y * img_h) for i in eye_indices]
    x, y, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    crop = image[y:y2, x:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    normalized = cv2.resize(gray, (NORM_SIZE, NORM_SIZE), interpolation=cv2.INTER_LINEAR)
    return cv2.Laplacian(normalized, cv2.CV_64F).var()
```

**Why the normalization step:** Laplacian variance scales with the pixel count of the region being scored. A tight headshot where eyes fill 30% of the frame and a wide-angle portrait where eyes fill 5% will produce non-comparable scores even when both are perfectly focused. Resizing to a fixed dimension before scoring decouples framing/sensor resolution from focus quality. Confidence this materially improves cross-photo comparability: **High**.

**Caveats:**
- High-ISO noise inflates scores. Mitigation: agent cross-checks EXIF ISO via `get_metadata`.
- Eye bbox < 20px wide before resizing → upscaling can't create sharpness information that wasn't there. Flag as low-confidence and skip rather than score.
- Even with normalization, cross-folder thresholds vary with lighting and lens; treat within-folder and within-burst-group as the trustworthy comparisons.

Confidence within-burst rankings are trustworthy (constant conditions): **High**. Confidence in cross-folder absolute thresholds: **Medium** — needs calibration per library.

Sources:
- Pech-Pacheco et al. (2000), Laplacian variance origin
- https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/

### No published end-to-end reference

The composition — MediaPipe face/eye landmarks → cropped Laplacian variance → portrait culling — is not published as a technique anywhere I could find. Consumer tools (Apple, Google, Adobe) do similar internally but don't publish methods. The pipeline composes well-validated components; it is not itself benchmarked. Validate empirically on real photos before committing.

Confidence: **Medium** the composed pipeline works well in practice. **High** that the individual components are sound.

---

## Burst-group flow

1. `find_burst_groups(folder)` → groups via pHash within the folder (Hamming ≤ 5, tunable)
2. `rank_burst_group(paths)` → ranked by eye-sharpness, EXIF deltas surfaced
3. `get_thumbnail(path, annotate_face=true)` for top 2–3 candidates
4. Claude visually compares thumbnails, picks keeper based on expression/eyes-open/pose
5. Returns "keeper: X; cull candidates: Y, Z, ..."; user acts

The split is the point: server handles measurable CV; Claude handles judgment.

Confidence pHash groups tight bursts correctly: **High**. For looser sequences with composition changes: **Medium** — threshold needs tuning.

Sources:
- https://github.com/JohannesBuchner/imagehash
- https://www.hackerfactor.com/blog/?/archives/432-Looks-Like-It.html

---

## Sharpness flow

1. `find_unsharp_subjects(folder)` → photos below relative threshold
2. `find_no_subject(folder)` → face-detection failures (likely environmental or back-of-subject)
3. `get_thumbnail(path, annotate_face=true)` for each candidate
4. Claude visually confirms or rejects each flag (intentional softness vs. missed focus)
5. Returns reviewed cull list

Confidence catching obvious focus misses: **High**. Distinguishing intentional softness from missed focus without vision: **Low**. With visual confirmation: **High** — which is the entire point of the vision-confirmation step.

---

## RAW handling

Paired JPG only for v1. Each RAW expects a sibling JPG with the same basename. EXIF read from JPG. Orphans surfaced via `find_orphans`. RAW preview extraction (rawpy) deferred to v2.

---

## Cache schema

```sql
CREATE TABLE photos (
  path TEXT PRIMARY KEY,
  mtime REAL, size INTEGER,
  face_count INTEGER,
  eye_sharpness_min REAL,
  eye_sharpness_max REAL,
  whole_image_sharpness REAL,
  fallback_used INTEGER,
  phash TEXT,
  face_bboxes TEXT,                 -- JSON
  eye_bboxes TEXT,                  -- JSON
  iso INTEGER, shutter TEXT, aperture REAL, focal_length REAL,
  camera TEXT, taken_at TEXT, indexed_at TEXT
);
CREATE INDEX idx_photos_dir ON photos(path);
```

Index time estimate for 10K photos: **60–180 minutes**, dominated by MediaPipe + Laplacian. Confidence: **Medium**, varies with disk speed and CPU. Benchmark on 100 files first.

---

## JD mapping

| JD requirement | How this hits it | Strength |
|---|---|---|
| MCP | Server built from scratch, 12 tools | Strong |
| Advanced prompt engineering | Documented iterations across two flows | Medium (manual) |
| Agent development | Multi-step CV → vision orchestration | Strong |
| Evaluation frameworks | Manual precision/recall on 200 hand-labeled photos | Medium (honest) |
| Transcript analysis | N/A | — |
| Deployment at scale | Single-user local | Weak |
| Python production code | Yes, with non-trivial CV | Strong |
| LLM + vision integration | Server returns annotated images; Claude confirms server's CV judgments | **Strong, rare in demo projects** |

The vision-confirmation pattern is the most differentiated piece. It's a real production pattern (visual QA, content moderation, medical imaging review) applied to a personal-use case.

---

## Stack

| Component | Library | Confidence |
|---|---|---|
| MCP server | `mcp` Python SDK (FastMCP) | High |
| Face / eye landmarks | `mediapipe` (requires `face_landmarker.task` download) | High |
| Image I/O | `Pillow` | High |
| CV ops | `opencv-python` (headless) | High |
| Perceptual hash | `imagehash` | High |
| Cache | `sqlite3` (stdlib) | High |
| Client | Claude Desktop (Pro) | High |
| Dep management | `uv` | High |

Install footprint ~250MB (MediaPipe brings TF Lite). All local, free.

---

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Eye-region Laplacian doesn't track subjective judgment | Medium | Two-evening prototype on 100 photos before full build |
| MediaPipe detection rate too low on your shooting style | Medium | Same prototype catches this; fallback path already designed |
| pHash burst threshold needs per-style tuning | High | Tunable parameter; default 5; calibrate on real bursts |
| 10K-photo index takes hours, feels bad | Medium | Threading, progress reporting, checkpoint writes |
| MCP image-return format has undocumented edge cases | Low-Medium | Test `get_thumbnail` on evening 3, before building dependent tools |

---

## Sources

**MCP**
- Python SDK: https://github.com/modelcontextprotocol/python-sdk (High)
- Build-server tutorial: https://modelcontextprotocol.io/docs/develop/build-server (High)
- Spec / image content in tool returns: https://modelcontextprotocol.io/specification (Medium — verify current SDK)

**Face detection**
- MediaPipe Face Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker (High)
- Face detection library comparison: https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/ (Medium-High, dated)

**Sharpness**
- Pech-Pacheco et al. 2000 (origin) (High)
- PyImageSearch walkthrough: https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/ (High)
- OpenCV Laplacian docs: https://docs.opencv.org/4.x/d5/db5/tutorial_laplace_operator.html (High)

**Perceptual hashing**
- imagehash: https://github.com/JohannesBuchner/imagehash (High)
- pHash background: https://www.hackerfactor.com/blog/?/archives/432-Looks-Like-It.html (High)

**Image I/O**
- Pillow: https://pillow.readthedocs.io/ (High)
- rawpy (v2 territory): https://letmaik.github.io/rawpy/ (Medium-High)

**Composed pipeline (face → eye crop → Laplacian → portrait culling):** no clean reference. Confidence Medium pending evening-2 validation.
