# DeepFoul Progress

Claude Code updates this file at the end of every ROADMAP section. Newest entries at the top of the log.

## Current status
| Phase | Section | Status |
|---|---|---|
| 0 | 0.1 Repo and environment | Done (RTX 2080) |
| 0 | 0.2 Starter fights | Done (15/15 downloaded) |
| 1 | 1.1 Video IO and shot detection | Not started |
| 1 | 1.2 Detection and tracking | Not started |
| 1 | 1.3 Identity: ref filter and red/blue | Not started |
| 1 | 1.4 Pose extraction and caching | Not started |
| 1 | 1.5 Overlay renderer v1 + benchmark | Not started |
| 2 | 2.1 Labeling app | Not started |
| 2 | 2.2 Stoppage finder | Not started |
| 2 | 2.3 Dataset builder | Not started |
| 3 | 3.1 Evaluation harness | Not started |
| 3 | 3.2 Feature layer | Not started |
| 3 | 3.3 Rule detectors v0 | Not started |
| 4 | 4.1 Error-driven fixes | Not started |
| 4 | 4.2 Replays and broadcast junk | Not started |
| 5 | 5.1 Baseline model | Not started |
| 5 | 5.2 Temporal model | Not started |
| 5 | 5.3 Hard-negative loop | Not started |
| 6 | 6.1 Final test run | Not started |
| 6 | 6.2 Overlay v2 | Not started |
| 6 | 6.3 Stats graphic + live-readiness | Not started |
| 7 | 7.1 Repo polish | Not started |

## Key numbers
- Pipeline speed (fps on RTX 2080): -
- Labeled positives: eye_poke - / low_blow - / open_hand -
- Fights: train 9 / val 4 / test 2 (8 foul, 5 clean, 2 test). Known fouls: 16 (`labels/notes/known_fouls.csv`)
- Best val metrics: -
- Test metrics: -

## Open issues
- openmmlab's model server throttles to ~100 KB/s per connection. On a new machine, pre-download the two model zips from rtmlib's Hugging Face mirror (see the 0.1 log entry).
- **6 unofficial uploads** (Aspinall-Gane, Battle-Loosa, Garbrandt-Long, Holland-Malott, Munhoz-O'Malley, Edwards-Muhammad). Accepted for now; ask Cowork for official replacements in a data round. They can vanish or be re-uploaded with a different cut, which would break every timestamp for that fight. If a URL is swapped, the fight's window and known fouls have to be re-timed.
- Edwards-Muhammad is 480p (under the 720p bar). Kept for train; drop it if pose is poor in 1.4.
- Garbrandt-Long is a Russian broadcast at 1694x1080, 60 fps (probably cropped and frame-interpolated). Munhoz-O'Malley is 50 fps.
- Jandiroba-Dern has a sponsor overlay (bottom right) burned into the official UFC upload.
- No Gane-Hokit (UFC 334) row yet; add it once a video exists. No similar-shorts fight confirmed yet.

## Switching machines (later, not now)
All work is on the RTX 2080 PC for now. When Ari moves to the 4090 PC (or any new Windows PC):
1. Turn off Smart App Control if it's on (it blocks unsigned DLLs).
2. Set user env var `UV_PYTHON_INSTALL_DIR=C:\Users\AriSh\.uv-python`.
3. Install uv, ffmpeg, yt-dlp and gh via winget; `gh auth login`.
4. `git clone` / `git pull`, then `uv sync`.
5. Pre-download the rtmlib model zips from the Hugging Face mirror (see the 0.1 log entry).
6. `uv run deepfoul doctor` must show `CUDAExecutionProvider` active. If it fails, pin onnxruntime-gpu to its CUDA 12 build before anything else.
7. `deepfoul fetch` to rebuild `data/raw/` from `labels/fights.csv`.

## Log

### 2026-11-19 · Section 0.2 · Starter fights
**Built:**
- `labels/fights.csv`: 15 starter fights with `action_start_s` / `action_end_s` (from Ari's windows, in seconds). The `fights_starter.*` files are removed.
- `labels/notes/starter_fights.md`: Ari's known fouls (live and replay times), corners, sources, and the dropped fights with reasons.
- `labels/notes/known_fouls.csv`: the 16 known fouls in seconds (`fight_id, live_s, replay_s, foul_type, offender, victim, ref_paused, visibility, notes`), for 2.2 and the golden set. 2 of them are in test fights, so filter by split.
- `src/deepfoul/io/fights.py`: `FIGHT_COLS`, `parse_mmss`, `window`, `validate_fights` (fight / `comp_<name>_<nn>` / `clip_<fight id>` ids, splits, dates, YouTube URLs, windows). It's meant for reuse by `ingest` in 2.3.
- `deepfoul fetch-check` (`src/deepfoul/io/probe.py`): ffprobe stats and flags, a contact sheet per fight at 10/50/90% of the fight window, and `outputs/fetch_check/report.md` with a summary per bucket.
- 33 new tests (43 total), including: the committed fights.csv is valid, and every known foul is inside its window.

**Decisions:**
- I validated the CSV while planning (yt-dlp metadata) and asked Ari then, so there was no mid-session pause. Ari accepted the 6 unofficial uploads (flagged for replacement later), kept the 480p Edwards-Muhammad, and skipped a Gane-Hokit placeholder.
- The duration flag only applies to rows without a window. With a window, a video is flagged only if it's shorter than `action_end_s` (Jones-Gane is short on purpose).
- Contact sheets use the fight window, not the whole video, so they show fighters instead of walkouts.
- `known_fouls.csv` got an extra `notes` column for things like "clear in replay at 09:28". Gaethje-Holloway poke 1 ("clearer in replay") is marked `hard_live`.

**Verified:**
- `deepfoul fetch`: 13 downloaded on the first pass. Garbrandt-Long and Holland-Malott hit a YouTube HTTP 403 and succeeded on one retry with `--only`. **15/15 downloaded.**
- `deepfoul fetch-check`: no length or frame rate problems. Flags: 6 unofficial uploads, and Edwards-Muhammad under 720p. Per bucket: test 2/2, foul 8/8, clean 5/5 downloaded.
- I looked at 6 contact sheets (Garbrandt, Edwards, Jandiroba-Dern, Munhoz-O'Malley, Holland-Malott, Aspinall-Gane): right fighters, full broadcast.
- `git status` showed nothing under `data/` or `outputs/`. `uv run pytest`: 43 passed.

**Golden set:** n/a.

**Known issues:**
- See Open issues (unofficial uploads, 480p, Russian broadcast, sponsor overlay).
- The windows include between-round segments, e.g. Munhoz-O'Malley 08:13 is a split-screen of the corners. These are handled in 4.2.
- **For 1.3:** in Holland-Malott, red-corner Holland wears **black** and blue-corner Malott wears **red** (the broadcast "trunk colors" graphic shows it). Shorts color ≠ corner color.
- Cosmetic: the timestamp on a contact sheet can overlap a channel watermark in the top left.

**Next:**
- Ari: open `outputs/fetch_check/report.md` and check every contact sheet (5 s each). The fight windows are already in `fights.csv`, so step 4 of the 0.2 "You do" and the "Fight windows" line in the 1.1 kickoff aren't needed.
- Then section 1.1 (video IO and shot detection).

### 2026-11-18 · Section 0.1 · Repo and environment
**Built:**
- uv project (Python 3.11.18), repo layout from CLAUDE.md, `.gitignore`, Apache-2.0 `LICENSE`, one-line `README.md`
- `configs/default.yaml` (paths, VRAM-based profiles, placeholder thresholds) and `src/deepfoul/config.py`
- `labels/{fights,events,golden}.csv` (headers only)
- CLI (`src/deepfoul/cli.py`):
  - `deepfoul doctor`: OS, Python, GPU/VRAM, onnxruntime providers, a real CUDA session test, profile, ffmpeg/yt-dlp versions
  - `deepfoul fetch [--csv] [--only]`: yt-dlp to `data/raw/<fight_id>.mp4`, ≤1080p, skips existing files
  - `deepfoul pose-check <image>`: rtmlib Wholebody to `outputs/pose_check.jpg`; a temporary helper until 1.4
- 10 pytest tests
- Machine setup on the 2080 PC:
  - installed uv 0.13.9, ffmpeg 9.0.2, yt-dlp 2026.10.26 and gh 2.105 via winget (git was already there)
  - set `attribution: {commit: "", pr: ""}` in `~/.claude/settings.json` and the committed `.claude/settings.json`
  - set the repo's git email to Ari's GitHub noreply address

**Decisions:**
- CUDA and cuDNN come from pip wheels (`onnxruntime-gpu[cuda,cudnn]` + `preload_dlls()`), with no system CUDA install, so every machine gets matching versions.
- onnxruntime-gpu 1.32 (CUDA 13 build) works on the 2080, so no CUDA 12 pin was needed.
- uv `override-dependencies` blocks the CPU `onnxruntime` that rtmlib pulls in (it would overwrite the GPU package) and `opencv-python` (it clashes with rtmlib's `opencv-contrib-python` over `cv2`).
- `required-environments` (win AMD64, mac arm64) makes uv pick NVIDIA wheels that exist for Windows; the newest nvidia-cublas is Linux-only.
- Both GPU profiles use pose `performance` (RTMW-x 384x288 + YOLOX-m); only batch size differs (large 16, small 4). `lightweight` is CPU only.
- Doctor builds its tiny test ONNX model by hand instead of depending on the `onnx` package.
- fetch reads CSVs as `utf-8-sig`, because Excel and PowerShell add a BOM that silently broke the `fight_id` column.
- `python -m deepfoul` also works (`__main__.py`), as a fallback if the `deepfoul.exe` launcher is ever blocked.

**Verified (RTX 2080, 8 GB):**
- `uv run deepfoul doctor`: CUDA session test OK, `CUDAExecutionProvider` active, profile `small`.
- `deepfoul fetch` on a scratch one-row CSV downloaded Stirling vs Blachowicz (UFC official channel, 1080p30, 4:40) to `data/raw/`. A re-run skipped it.
- `deepfoul pose-check` on a frame at 1:30: det and pose both on `CUDAExecutionProvider`, 2 people, keypoints (2, 133, 2). Skeleton image checked by eye: body, hands, face and feet all line up. The ref (partly hidden) was not detected; that's for 1.2.
- `git status` after the download shows nothing under `data/` or `outputs/`. `uv run pytest`: 10 passed.
- No YouTube JS runtime (Deno) was needed.

**Golden set:** n/a (no golden set yet).

**Known issues / machine notes:**
- **Smart App Control** blocked unsigned DLLs (uv Python's OpenSSL, `onnx`, the `deepfoul.exe` launcher). Ari turned SAC off on the 2080 PC. Expect the same on any other Windows PC with SAC on.
- **Claude desktop app redirects `AppData\Roaming`** for processes it starts, so uv's Python landed in the app's private folder and the venv didn't work from Ari's own terminal. Fix: user env var `UV_PYTHON_INSTALL_DIR=C:\Users\AriSh\.uv-python`, then reinstall 3.11 and rebuild `.venv`. Needed on any Windows PC used through the Claude desktop app.
- Model download: openmmlab is ~100 KB/s. The maintainer's mirror `huggingface.co/Tau-J/RTMPose` has the same files (sizes and sampled byte ranges checked against openmmlab). Put the zips in `~/.cache/rtmlib/hub/checkpoints/` and rtmlib unpacks them on first use.
- Leftovers from the first (redirected) Python install: `C:\Users\AriSh\.local\bin\python3.11.exe` and a copy under the Claude app's LocalCache. Both are unused and safe to delete.
- `labels/fights_starter.csv` and `fights_starter_notes.md` appeared during this session. They weren't made here (they look like the starter-list research), so they're left uncommitted for Ari to review.

**Next:**
- Ari ran `gh auth login`: create `ar1shah/deepfoul` and push.
- Ari: review the starter fight list.
- Then section 1.1 (video IO and shot detection).

<!--
Template for each entry:

### YYYY-MM-DD · Section X.Y · <name>
**Built:** what exists now (files, commands)
**Decisions:** what was chosen and why (one line each)
**Verified:** how the Done-when criteria were checked, with results
**Golden set:** pass / regressions
**Known issues:** anything flaky or deferred
**Next:** the next section, plus anything Ari needs to do first
-->
