# Manual Test Plan — Evenings 1–4

---

## Evening 1 — score_portraits.py

```bash
uv run python score_portraits.py temp/ scores_manual.csv
```

- [ ] 175 rows in CSV (plus header)
- [ ] Columns: `path, face_count, eye_sharpness_min, eye_sharpness_max, whole_image_sharpness, fallback_used`
- [ ] Spot-check 3 rows with `face_count > 0` — open those photos, confirm visible faces
- [ ] Spot-check 3 rows with `fallback_used=True` — confirm no clear frontal face
- [ ] `eye_sharpness_min` ≤ `eye_sharpness_max` wherever both are present

---

## Evening 3 — server.py (via Claude Desktop)

Restart Claude Desktop first. Confirm "triage-keepers" shows as connected.

**list_folders**
```
Call list_folders with root="/Users/ak/Documents/coding/photography_code/triage-keepers"
```
- [ ] Returns `.git`, `.venv`, `temp` among results
- [ ] No files in result, only directories

**get_thumbnail**
```
Call get_thumbnail with path="<any JPG from temp>"
```
- [ ] Image renders inline in Claude Desktop (not a raw base64 string)
- [ ] Thumbnail is visibly smaller than full resolution

**get_thumbnail — bad path**
```
Call get_thumbnail with path="/nonexistent.jpg"
```
- [ ] Returns an error, server does not crash

---

## Evening 4 — cache.py + index_folder

**First run (fresh DB)**
```bash
rm -f ~/.triage-keepers/cache.db
```
```
Call index_folder with path="<temp folder path>"
```
- [ ] Returns `{"total": 175, "indexed": 175, "skipped": 0}`

**Second run (cache hits)**
```
Call index_folder with path="<temp folder path>"
```
- [ ] Returns `{"total": 175, "indexed": 0, "skipped": 175}`

**DB persistence**
```bash
sqlite3 ~/.triage-keepers/cache.db "SELECT path, mtime, size, indexed_at FROM photos LIMIT 5;"
```
- [ ] 5 rows returned with non-null `path`, `mtime`, `size`, `indexed_at`

**File change detection**
```bash
touch /Users/ak/Documents/coding/photography_code/triage-keepers/temp/IMG_4223.JPG
```
```
Call index_folder with path="<temp folder path>"
```
- [ ] Returns `{"total": 175, "indexed": 1, "skipped": 174}`
