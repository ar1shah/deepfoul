# DeepFoul Progress

Claude Code updates this file at the end of every ROADMAP section. Newest entries at the top of the log.

## Current status
| Phase | Section | Status |
|---|---|---|
| 0 | 0.1 Repo and environment | Done on RTX 2080; 4090 doctor check pending |
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
- Pipeline speed (fps on 4090): -
- Labeled positives: eye_poke - / low_blow - / open_hand -
- Fights: train - / val - / test -
- Best val metrics: -
- Test metrics: -

## Open issues
- Run `uv run deepfoul doctor` on the 4090 PC (needs a driver new enough for CUDA 13). If CUDA fails there, pin onnxruntime-gpu to its CUDA 12 build before anything else.
- openmmlab's model server throttles to ~100 KB/s per connection. On a new machine, pre-download the two model zips from rtmlib's Hugging Face mirror (see the 0.1 log entry).

## Log

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
- Not done: the 4090 doctor check (still open). No YouTube JS runtime (Deno) was needed.

**Golden set:** n/a (no golden set yet).

**Known issues / machine notes:**
- **Smart App Control** blocked unsigned DLLs (uv Python's OpenSSL, `onnx`, the `deepfoul.exe` launcher). Ari turned SAC off on the 2080 PC. Expect the same on the 4090 and other Windows PCs if SAC is on.
- **Claude desktop app redirects `AppData\Roaming`** for processes it starts, so uv's Python landed in the app's private folder and the venv didn't work from Ari's own terminal. Fix: user env var `UV_PYTHON_INSTALL_DIR=C:\Users\AriSh\.uv-python`, then reinstall 3.11 and rebuild `.venv`. Do the same on the 4090 PC.
- Model download: openmmlab is ~100 KB/s. The maintainer's mirror `huggingface.co/Tau-J/RTMPose` has the same files (sizes and sampled byte ranges checked against openmmlab). Put the zips in `~/.cache/rtmlib/hub/checkpoints/` and rtmlib unpacks them on first use.
- Leftovers from the first (redirected) Python install: `C:\Users\AriSh\.local\bin\python3.11.exe` and a copy under the Claude app's LocalCache. Both are unused and safe to delete.
- `labels/fights_starter.csv` and `fights_starter_notes.md` appeared during this session. They weren't made here (they look like the starter-list research), so they're left uncommitted for Ari to review.

**Next:**
- Ari: run `gh auth login`, create `ar1shah/deepfoul`, then push.
- Ari: review the starter fight list.
- On the 4090 PC: `git pull`, set `UV_PYTHON_INSTALL_DIR`, `uv sync`, `uv run deepfoul doctor`.
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
