# DeepFoul

Computer vision system that watches recorded MMA fight footage and flags fouls for human review. v1 covers **eye pokes** and **low blows**, plus **open hands** (fingers extended toward the opponent's face) as an early-warning signal for eye pokes.

It flags clips for a replay official. It never makes the call itself (think VAR, not goal-line tech).

## Why this exists
- Fouls in MMA depend on whether the ref happens to see them. Eye pokes alone happen roughly once every 14 UFC fights and regularly end fights in no-contests.
- The project is open source and is also a portfolio piece built to pitch to Jabbr (makers of DeepStrike, a combat sports CV system for punch stats and live scoring). DeepFoul is designed as a module that could plug into a system like DeepStrike: same red/blue corner convention, same "event log" style output.
- DeepFoul is NOT affiliated with Jabbr. Never use their logo, the DeepStrike name, or their branding in overlays, assets or docs (except plain-text mentions in the pitch/README context).

## Owner and workflow
- Owner: Ari Shah (GitHub: ar1shah). Ari makes product decisions and does all data collection and labeling.
- Work happens **one ROADMAP section at a time**. For each section:
  1. Read `CLAUDE.md`, `ROADMAP.md` and `PROGRESS.md` first.
  2. Propose a plan for the section (files, approach, open questions). Wait for Ari's approval before writing code.
  3. When a decision is Ari's to make, give 2 to 3 options with a recommendation and the tradeoff in one line each.
  4. Build it, run it, and verify the section's **Done when** criteria for real (on actual clips, not just unit tests).
  5. Update `PROGRESS.md` (what was built, decisions and why, known issues, next step). Then stop.
- Do not start the next section on your own.
- If a section turns out much bigger than expected, stop and propose splitting it.

## Environment
- Work moves between machines, so nothing can be machine-specific:
  - Windows desktop with an RTX 4090 (24 GB): main GPU machine
  - Windows gaming PC with an RTX 2080 (8 GB): GPU machine while Ari is home. Everything must run within 8 GB of VRAM (configurable batch sizes, model size presets).
  - Mac: development and labeling only; CPU fallback for quick tests on short clips
- Native Windows on the PCs, not WSL, unless dependencies force a switch (ask first).
- The code syncs through GitHub. Footage never syncs: `deepfoul fetch` re-downloads every video listed in `labels/fights.csv` into `data/raw/`, so any machine can rebuild its data from the CSV.
- `deepfoul doctor` reports which machine profile it detects (GPU name, VRAM, provider) and the config picks sensible defaults from that.
- Python 3.11, managed with `uv` (`uv sync`, `uv run ...`). Dependencies in `pyproject.toml`.
- GPU inference through `onnxruntime-gpu`. Make sure the CUDA/cuDNN versions match the onnxruntime-gpu build; `deepfoul doctor` must confirm `CUDAExecutionProvider` is active.
- External tools: `ffmpeg` and `yt-dlp` on PATH.
- Use `pathlib` everywhere. No hardcoded absolute paths; everything configurable via `configs/*.yaml`.

## Stack (all open source, commercial-friendly licenses)
| Job | Tool | License |
|---|---|---|
| Whole-body pose (133 keypoints) | RTMW via `rtmlib` (ONNX) | Apache 2.0 |
| Person detection | YOLOX detector bundled with `rtmlib` | Apache 2.0 |
| Tracking | ByteTrack (via `supervision`) | MIT |
| Drawing overlays | `supervision` + OpenCV | MIT / Apache 2.0 |
| Shot / camera-cut detection | PySceneDetect | BSD-3 |
| Classifier | LightGBM baseline, then a small PyTorch temporal model | MIT / BSD |
| Labeling app | FastAPI + plain HTML/JS, local only | MIT |

**Check current docs before writing library code.** rtmlib, supervision, onnxruntime and PySceneDetect change their APIs often. Before using one of their APIs (especially in a plan), look it up with the Context7 MCP. If Context7 doesn't cover the library, read its GitHub README or source. Don't write these APIs from memory.

Do not add AGPL or non-commercial dependencies (e.g. Ultralytics YOLO, OpenPose) without asking.

## Architecture (two halves)
**Perception (shared by training and detection)**, built once, used twice:
1. Video in -> split into shots at camera cuts.
2. Detect people -> track -> build per-fight color profiles -> drop the ref, crowd and cornermen -> assign `red` / `blue` corner.
3. RTMW pose on each fighter's box -> 133 keypoints + confidence per frame.
4. Cache keypoints to `data/keypoints/<fight_id>.parquet` so pose runs only once per video.

**Training side**: stoppage finder (mines candidate foul moments), labeling app, dataset builder, model training, evaluation.

**Detector side**: features -> rules or trained classifier scores sliding windows -> event logic -> flagged events with confidence -> overlay video + stats.

### Keypoint reference (COCO-WholeBody 133, verify against rtmlib docs before relying on it)
- 0 nose, 1/2 L/R eye (body set), 5/6 shoulders, 9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles
- 17 to 22 feet (big toe, small toe, heel per side)
- 23 to 90 face (68 landmarks; eye contours are face points 36 to 47, i.e. indices 59 to 70)
- 91 to 111 left hand, 112 to 132 right hand (21 each; offset 0 wrist, 4 thumb tip, 8 index tip, 12 middle tip, 16 ring tip, 20 pinky tip)
- Classifier features use a ~30 point subset (fingertips, wrists, eyes, nose, shoulders, hips, knees, ankles, feet). Keep the full 133 in the cache.

## Design rules
- **Streaming-friendly / live-ready.** Process frames in order. A decision about time `t` may only use frames up to `t + lookahead`, where `lookahead` is a fixed config value (default 2.0 s, needed to see the reaction). No other peeking into the future. This keeps a later live version a matter of swapping the input source.
- **Flags, not verdicts.** Every output event has a type, offender corner, time range and confidence.
- **Normalize by body scale** (e.g. torso length) so features work at any zoom level.
- **Respect keypoint confidence.** Low-confidence points are masked, never trusted silently.
- **Split by fight, never by clip.** A fight's clips all live in one split. Test fights are never used for training or tuning.
- Keep it simple. No abstractions until there is a second user of them.

## Data rules (important)
- `data/` and `outputs/` are gitignored. **Never commit video, clips, frames or downloaded footage.** Check `git status` before every commit.
- What IS committed: `labels/fights.csv`, `labels/events.csv`, `labels/golden.csv` (metadata only: URLs, timestamps, labels) and trained model weights (via GitHub Releases, not the repo, if large).
- Held-out test set: Ciryl Gane fights (vs Tom Aspinall at UFC 321, vs Josh Hokit at UFC 334, plus any others Ari adds). These are `split=test` in `fights.csv` and are only run at the end of a phase for reporting.

### Label schemas
`labels/fights.csv`: `fight_id, event, date, red_fighter, blue_fighter, red_shorts, blue_shorts, source_url, split (train|val|test), notes`

`labels/events.csv`: `event_id, fight_id, start_s, end_s, foul_type (eye_poke|low_blow|open_hand|hard_negative|clean), offender (red|blue|none), contact (yes|no|unsure), ref_stopped (yes|no), speed (live|slowmo), source (full_fight|compilation|mined), notes`

`labels/golden.csv`: same columns as events.csv, for the ~5 fixed regression clips.

## Evaluation conventions
- Event-level matching: a predicted flag counts as a hit if it overlaps a labeled event of the same type within a tolerance (default ±1.0 s).
- Report per foul type: recall (fouls caught), precision (flags that were real), false flags per fight. Never report plain accuracy.
- Every section that touches perception or detection re-runs the golden set and notes regressions in PROGRESS.md.

## Repo layout
```
deepfoul/
  CLAUDE.md  ROADMAP.md  PROGRESS.md  README.md  LICENSE (Apache-2.0)  pyproject.toml
  configs/            default.yaml (thresholds, windows, paths)
  labels/             fights.csv, events.csv, golden.csv   (committed)
  data/               raw/ clips/ keypoints/ cache/        (gitignored)
  outputs/            renders, reports, benchmarks         (gitignored)
  src/deepfoul/
    io/               video reading, shot detection
    perception/       detect, track, identity (ref filter, red/blue), pose
    features/         keypoint subset, normalization, zones, kinematics
    rules/            open_hand, eye_poke, low_blow
    events/           window scores -> events (hysteresis, merging, cooldown)
    mining/           stoppage finder
    labeling/         local labeling app
    dataset/          build training windows from labels
    models/           train / infer
    eval/             metrics, reports
    viz/              overlays, stats graphics
    cli.py            `deepfoul <command>`
  tests/
```

CLI commands (added as phases land): `doctor, extract, render, mine, label, build-dataset, detect, train, eval, bench`.

## Git
- Commits are authored by Ari only. **Do not add `Co-Authored-By` trailers, "Generated with Claude Code" lines, or any AI attribution** to commits or PRs.
- Small, focused commits with short imperative messages (e.g. `add ByteTrack tracking with ref filter`).
- Never commit secrets, footage, or anything under `data/` or `outputs/`.

## Testing
- `pytest` for logic that can be tested without video: geometry, zones, rules on synthetic keypoint sequences, event merging, metrics, label parsing.
- Real-footage checks happen through the golden set and `deepfoul render` outputs that Ari reviews by eye.
