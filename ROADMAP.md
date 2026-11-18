# DeepFoul Roadmap

Each section is sized for one Claude Code session. Every section has:
- **Build**: everything Claude Code does in that session. That includes code, plus scripted chores like installs, downloads, batch runs, renders and review reels. If a chore can run from the repo, it belongs here.
- **Done when**: the concrete check before moving on.
- **Ari**: work outside the Claude Code session, split into:
  - **Claude chat / Cowork**: research and writing done separately from Claude Code (finding fights and URLs, drafting posts and emails), plus check-ins on plans and results. Results go into the repo (Cowork can write them directly) or get pasted into Claude Code.
  - **You do**: things only Ari can do: anything needing his accounts or logins, judgment calls, and **all labeling** (ground truth must come from a human, never from Claude).

Ari's data work runs in parallel with the code. Start collecting footage on day 1; don't wait for the tools.

---

## Footage guide (read this first)

### Three kinds of footage
1. **Full fights**: the main source. Used for tracking, mining stoppages, negatives and most positives. Every full fight gets a row in `labels/fights.csv`.
2. **Foul compilations**: YouTube "eye poke / low blow compilation" videos. Only used to top up positives. Each gets a `fights.csv` row too (e.g. `comp_eyepokes_01`, `split=train`).
3. **Test fights**: Ciryl Gane's fights. Held out, never trained or tuned on.

### Quality bar for any video
- Full fight at **normal speed**, 720p minimum (1080p preferred)
- Official uploads first (the UFC's own YouTube channel posts free full fights)
- No reaction channels, commentary overlays, mirrored or cropped re-uploads
- Mostly UFC broadcasts from 2022 onward, so the cameras and graphics look consistent
- Footage stays in `data/raw/`, private, never redistributed

### Starter list for Phase 0 (15 to 20 full fights)
| Bucket | How many | What | Split |
|---|---|---|---|
| Test | 2 to 3 | Aspinall vs Gane (UFC 321), Gane vs Hokit (UFC 334), plus any other recent Gane fight with fouls | `test` |
| Foul fights | 8 to 10 | Fights where an eye poke or low blow visibly paused or ended the action. Roughly half eye pokes, half low blows. Known eye poke examples: Edwards vs Muhammad 1 (2021), O'Malley vs Munhoz (UFC 276), Weidman vs Bruno Silva. | 2 to 3 of these `val`, rest `train` |
| Clean fights | 5 to 7 | No notable fouls, mostly stand-up striking | 1 `val`, rest `train` |

**Variety rules for the starter list:**
- At least 2 women's fights (sports bras matter for the ref filter)
- At least 1 fight where both fighters wear similar or dark shorts
- A mix of striking-heavy and grappling-heavy fights
- A mix of weight classes (body size changes the geometry)

### Growth targets (Phases 2 to 5)
- 30+ train fights, 5 to 6 val fights. The val fights must contain at least 8 real fouls between them, or the val metrics mean nothing.
- 150+ eye pokes, 150+ low blows, 200+ hard negatives (details in 2.3).
- **Compilation leakage rule:** compilations often include famous pokes, including Gane's. Any compilation segment from a test fight gets skipped during labeling.

---

## Phase 0: Setup

### 0.1 Repo and environment
**Build**
- System setup: install `uv`, `ffmpeg`, `yt-dlp` and `gh` (winget) if missing, confirm `nvidia-smi`, install Python 3.11 through `uv`.
- Turn off co-author attribution in Claude Code settings (user settings on this machine, plus a committed project `.claude/settings.json` so it follows the repo to every machine).
- Set this repo's `git user.email` to Ari's GitHub noreply address before the first commit (the repo is public).
- `uv` project, Python 3.11, `pyproject.toml` with core deps (rtmlib, onnxruntime-gpu, opencv-python, supervision, scenedetect, pandas, pyarrow, pyyaml, typer, pytest). Make sure rtmlib can't pull in CPU `onnxruntime` next to `onnxruntime-gpu`.
- Repo layout from CLAUDE.md, `.gitignore` covering `data/`, `outputs/`, video extensions (`*.mp4 *.mkv *.webm *.mov`), `.venv/`.
- Apache-2.0 `LICENSE`, placeholder `README.md` (one line: what it is, "not affiliated with Jabbr").
- `configs/default.yaml` with paths, machine profiles and placeholder thresholds.
- `deepfoul doctor`: prints Python, onnxruntime providers (must include `CUDAExecutionProvider`), ffmpeg and yt-dlp versions, GPU name, selected profile.
- `labels/` CSVs created with headers only.
- `deepfoul fetch`: downloads every video in `fights.csv` with `yt-dlp` into `data/raw/<fight_id>.mp4`, skipping ones already there.
- Once Ari approves the starter list in `fights.csv`, run `deepfoul fetch` on it (end of 0.1, or start of 1.1 if the list isn't ready yet).

**Done when**
- `uv run deepfoul doctor` shows CUDA active on the current GPU (the 4090 check happens whenever Ari is next on that PC).
- rtmlib whole-body runs on one test image on GPU and saves an image with skeletons drawn.
- `git status` after downloading a test video shows nothing under `data/`.

**Ari**
- **Claude chat / Cowork:** research the starter list from the Footage guide. Find fights with documented eye pokes and low blows from news coverage, find official full-fight URLs, and note roughly when each foul happens. Output: rows for `labels/fights.csv` (Cowork can write them into the file directly).
- **You do:**
  - Run `gh auth login` and approve creating the GitHub repo `ar1shah/deepfoul` (public).
  - Give Claude Code your GitHub noreply email (GitHub > Settings > Emails).
  - Approve the fight list. Make sure each "foul fight" really shows the foul on camera, and swap out any fight that isn't available as a full, normal-speed video.
  - Spot-check that each download plays and is the full fight.

---

## Phase 1: Perception pipeline (the "eyes")

### 1.1 Video IO and shot detection
**Build**
- Frame reader that yields frames in order with timestamps, supports start/end times and frame stride.
- PySceneDetect shot boundaries, saved per fight. Tracks reset at cuts.
- `deepfoul extract --shots <video>` writes `data/cache/<fight_id>_shots.json`.
- A contact sheet (frames before and after 20 detected cuts) so Ari's spot-check takes 2 minutes.

**Done when**
- On 2 full fights, shot cuts land on real camera changes (Ari spot-checks the contact sheet).

**Ari**
- **Claude chat / Cowork:** propose golden-set candidates: approximate timestamps of the fouls in the foul fights, from news reports and fight recaps.
- **You do:**
  - Review the contact sheet.
  - Build the **golden set** in `labels/golden.csv`. Use **train fights only**, 10 to 30 s each, live speed. Find the exact times by watching:
    1. Clean stand-up exchange, both fighters fully visible
    2. Clinch against the cage
    3. Ref stepping in between the fighters (a break or a stoppage)
    4. A clear eye poke
    5. A clear low blow
    6. (Bonus) a stand-up exchange from a women's fight

### 1.2 Detection and tracking
**Build**
- Person detection (rtmlib YOLOX) + ByteTrack within each shot.
- Filters: drop small boxes and boxes outside the cage area (crowd, camera crew). Configurable thresholds.
- Debug renders (boxes + track IDs) for every golden clip.

**Done when**
- On the golden clips, the 2 fighters and ref each keep a stable track ID through stand-up exchanges. Crowd is not tracked.

**Ari**
- **You do:** watch the debug renders and log every ID swap with a timestamp.

### 1.3 Identity: ref filter and red/blue
**Build**
- Per-fight color profiles: sample torso and leg regions from early frames, build HSV histograms per person.
- Ref detection: the person with shirt + long pants profile (configurable, plus a fallback "third person who never fights" heuristic).
- Corner assignment from `fights.csv` shorts colors; carry identity across shot cuts by profile matching.
- Output: per frame, `red` and `blue` boxes (or missing), ref box marked separately.
- Fill `red_shorts` / `blue_shorts` in `fights.csv`: grab a frame from each fight with both fighters and the broadcast name graphic visible, read off colors and corners, and save those frames to `outputs/` so Ari can verify. Flag fights where the two shorts colors are similar.

**Done when**
- Golden clips render with correct red/blue labels, no box on the ref, and identities survive at least the cuts in those clips.
- Works on a fight where both fighters wear similar or dark shorts.

**Ari**
- **You do:** verify the shorts colors and corners against the saved frames. A wrong corner here flips every label for that fight.

### 1.4 Pose extraction and caching
**Build**
- RTMW whole-body on each fighter's tracked box. Save per frame: corner, box, 133 keypoints, confidences.
- `deepfoul extract <video>` runs shots + tracking + identity + pose and writes `data/keypoints/<fight_id>.parquet`. Resumable if interrupted.
- Frame stride configurable (default: every frame up to 30 fps).
- Once it works, run extraction on every downloaded fight (overnight batch) and report any failures.

**Done when**
- One full 3-round fight extracts end to end without crashing.
- Fingertip and eye keypoints visibly track on the golden poke clip (check with 1.5's render).

**Ari**
- **You do:** leave the PC on overnight for the batch run.

### 1.5 Overlay renderer v1 + benchmark
**Build**
- `deepfoul render <video> --start --end`: red/blue boxes, corner labels, optional skeleton toggle, frame timestamp.
- `deepfoul bench`: reports processing speed (fps) for detection, pose, and full pipeline on the current GPU.
- Render the full golden set and run the benchmark.

**Done when**
- Golden set renders cleanly. Benchmark number is recorded in PROGRESS.md (per machine).

**Ari**
- **You do:** watch all golden renders. This is the first screenshot-worthy moment; post it if you're building in public.
- **Claude chat (check-in):** send the benchmark numbers and any recurring tracking failures before Phase 2.

---

## Phase 2: Data engine

### 2.1 Labeling app
**Build**
- Local FastAPI app + single HTML page. Loads a queue of candidate moments (fight_id + time), plays the clip around each one (default 4 s before, 4 s after), with playback speed control.
- Keyboard shortcuts: `1` eye poke, `2` low blow, `3` open hand, `4` hard negative, `5` clean, `R`/`B` offender corner, `C` contact yes/no, `S` skip, `←/→` scrub, adjust start/end with brackets.
- Writes rows to `labels/events.csv` immediately (no data loss on crash). Can also add a moment manually by timestamp.

**Done when**
- Ari labels 30 moments in under 10 minutes without touching the mouse much.

**Ari**
- **You do:** test it hard and write down anything slow or annoying. Most of your hours go into this tool, so it's worth polishing before the big labeling push.

### 2.2 Stoppage finder
**Build**
- Scans cached keypoints and flags candidate foul moments using signals that don't need a foul model:
  - Ref moves between or toward the fighters quickly
  - Both fighters' motion drops sharply (action paused)
  - A fighter's hand goes to his own eye region or groin region and stays
  - Fighter bent over at the waist
- Score each moment, output a ranked queue for the labeling app.
- `deepfoul mine <fight_id>` and `deepfoul mine --all`. Run it on every extracted fight and load the queues.

**Done when**
- On fights with known fouls, the real fouls appear in the top candidates (target: most known fouls in the top 20 per fight).

**Ari**
- **You do:** label the mined queues.
  - Real fouls -> `eye_poke` / `low_blow`
  - Stoppages for other reasons (cuts, ref warnings, slips, a glove touch) -> `clean` with a note. These double as hard negatives.

### 2.3 Dataset builder
**Build**
- Turns `events.csv` + cached keypoints into training windows (default 2 s window, configurable stride).
- Positives from labeled fouls; negatives sampled from clean time across fights, with extra weight on `hard_negative` rows.
- Train / val / test split by fight from `fights.csv`. Refuses to put any test fight in train or val.
- Summary report: counts per class per split.
- `deepfoul ingest`: one command for new fights (fetch -> extract -> mine), so adding fights during the data push is a single step.
- Hard-negative candidate queue: moments where a closed fist or a kick lands near a foul zone with no stoppage afterward, loaded into the labeling app.

**Done when**
- `deepfoul build-dataset` produces a dataset file and a summary, and a unit test proves no fight appears in two splits.

**Ari (the big data push, runs through Phases 2 to 4)**

Targets before Phase 5:

| Bucket | Target | Where it comes from |
|---|---|---|
| Eye pokes | 150+ | Mined from full fights first (live speed with context), then compilations |
| Low blows | 150+ | Same |
| Hard negatives | 200+ | Closed fists landing near the eye, jabs to the brow, inner-thigh leg kicks, clinch knees to the body, body kicks caught low |
| Clean | Automatic | Sampled by the dataset builder, no work for you |
| Full fights | 30+ train, 5 to 6 val | Keep adding using the Footage guide rules |

- **Claude chat / Cowork:** find more full fights with documented fouls, plus 4 to 6 eye poke and low blow compilations. Add them as rows in `fights.csv` (compilations as `split=train`). Then run `deepfoul ingest` in Claude Code.
- **You do:**
  - All labeling. Mark `speed=slowmo` on slow-motion replays and `source=compilation` on compilation clips.
  - Skip any compilation segment from a Gane fight (leakage rule).
- **Claude chat (check-in):** send the dataset summary when you hit ~50 positives per foul.

---

## Phase 3: Features, rules and evaluation

### 3.1 Evaluation harness (built before the rules so we can measure them)
**Build**
- Event matching (±1.0 s default), per-type recall, precision, false flags per fight.
- `deepfoul eval --split val` prints a table and saves a report with the list of misses and false flags (fight + timestamp) so Ari can watch them.

**Done when**
- Metrics are unit-tested on hand-made prediction/label lists, and a dummy "always flag" and "never flag" detector produce the expected numbers.

**Ari**
- **You do:** make sure every foul in the val fights is labeled. Any unlabeled foul turns into a fake "false flag" and drags the numbers down.

### 3.2 Feature layer
**Build**
- Keypoint subset, normalization by torso length, confidence masking.
- Cross-fighter geometry: each fighter's fingertips relative to the opponent's eye region; feet/shins/knees relative to the opponent's groin zone (below hip midpoint, between hips).
- Kinematics: velocities, head jerk, torso bend angle, hand-to-own-face and hand-to-own-groin distances.
- Finger extension score per hand (fingertip-to-wrist distance vs knuckle-to-wrist, plus joint straightness).
- Feature plots for the golden poke and low blow clips.

**Done when**
- The golden poke plot shows the fingertip-to-eye distance dropping at the poke and the reaction features spiking after it.

**Ari**
- **You do:** look at the plots and confirm the spikes line up with what you see in the clip. Keep labeling in the background.

### 3.3 Rule detectors v0
**Build**
- `open_hand`: fingers extended + pointing at the opponent's face + within range.
- `eye_poke`: fingertip enters the eye zone, then a reaction within the lookahead (hand to face, head snap, action stops).
- `low_blow`: foot/shin/knee enters the groin zone, then a reaction (bend at waist, hands to groin, action stops).
- Event logic: window scores -> events with hysteresis, merging of nearby flags, cooldown, offender corner.
- `deepfoul detect <video>` writes `outputs/<fight_id>_events.json`.
- A single review reel of every false flag and miss from the val eval report, with a caption on each (fight, time, predicted vs labeled).

**Done when**
- Rules fire on the golden poke and low blow clips and stay quiet on the clean clip.
- `deepfoul eval --split val` report exists with real numbers (they'll be rough; that's fine).

**Ari**
- **You do:** watch the review reel and sort each mistake into a bucket (tracking failure, keypoint jitter, camera angle, wrong threshold, label mistake).
- **Claude chat (check-in):** send the bucket counts. This decides what Phase 4 fixes.

---

## Phase 4: Make it not suck

### 4.1 Error-driven fixes
**Build**
- Fix the top 2 to 3 failure buckets from 3.3 (likely candidates: keypoint smoothing, better ref/ID handling in clinches, replay/slow-motion handling, depth ambiguity from 2D overlap).
- Tune thresholds on **val only**.

**Done when**
- Val metrics improve on the targeted buckets and the golden set has no regressions.

**Ari**
- **You do:** fix any labels the review reel showed were wrong. Keep labeling, prioritizing hard negatives that fooled the rules.

### 4.2 Replays and broadcast junk
**Build**
- Collect 10 to 15 example replay transitions from the downloaded fights as test cases.
- Detect broadcast replays (slow motion and/or the replay transition graphic) so the same foul isn't counted twice; replays get tagged, not flagged.
- Skip between-round segments (cornermen in frame).

**Done when**
- No double flags from replays on 3 full val fights.

**Ari**
- **You do:** confirm the replay examples are real replays. Keep labeling.

---

## Phase 5: Learned classifier

### 5.1 Baseline model
**Build**
- LightGBM on window-level engineered features (from 3.2). Class weights for imbalance.
- Thresholds picked on val. Compare against rules in one table.

**Done when**
- Comparison table (rules vs LightGBM) on val is in PROGRESS.md.

**Ari**
- **You do:** hit the 2.3 targets before this section starts. Under ~100 positives per foul, the model will just memorize.

### 5.2 Temporal model
**Build**
- Small PyTorch model (1D CNN or GRU) on normalized keypoint sequences per window, two-fighter input. Trains in minutes on either GPU.
- Simple augmentation: horizontal flip with left/right keypoint swap, small scale/jitter noise.
- `deepfoul train` and saved weights with config.

**Done when**
- Three-way comparison (rules, LightGBM, temporal) on val. Pick the default detector (or an ensemble, e.g. open_hand from rules + fouls from the model).

**Ari**
- **Claude chat (check-in):** make the pick together using the comparison table.

### 5.3 Hard-negative loop
**Build**
- Export the model's top-scoring false flags on train fights into the labeling queue.
- After Ari labels them, retrain and re-run the comparison.

**Done when**
- One full loop done: Ari labels the exported queue, retrain, val improves or the PROGRESS.md entry explains why not.

**Ari**
- **You do:** label the exported false flags (mostly `hard_negative`).
- **Claude chat (check-in):** send the comparison table before running the test set.

---

## Phase 6: The demo (detector as a product)

### 6.1 Final test run
**Build**
- Run the chosen detector on the held-out Gane test fights once. Save the report. No tuning after this run (if something is tuned, it has to be re-justified on val and noted).

**Done when**
- Test metrics and every flagged event (with timestamps) are in PROGRESS.md.

**Ari**
- **You do:** label every foul and open hand in the Gane test fights **before** this run (in one sitting, without looking at any model output). Then don't touch anything after the run.

### 6.2 Overlay v2 (broadcast style)
**Build**
- Red/blue boxes with head and torso sub-boxes per fighter.
- Foul zone highlight: the eye region or groin zone glows when a hand or foot enters it.
- On a flag: box flashes + banner (e.g. `EYE POKE · BLUE -> RED · 0.87 · REVIEW`).
- Skeleton shown only for ~1 s around a flag.
- Side panel: foul log (time, type, offender, confidence) and open hands per round.
- Own visual identity. No Jabbr/DeepStrike branding.

**Done when**
- Renders of the Gane pokes look good enough to post.

**Ari**
- **You do:** give design feedback on the renders (colors, panel layout, banner timing).

### 6.3 Stats graphic + live-readiness check
**Build**
- Per-fight shareable graphic (no footage): open hands per round per fighter, flagged fouls, timeline.
- Streaming mode check: run the detector frame-by-frame with only the allowed lookahead and confirm identical events to batch mode. Record latency and fps.
- Export short candidate clips of every flagged moment in the test fights.

**Done when**
- Graphic exported as PNG; streaming vs batch match; fps + latency numbers in PROGRESS.md (this becomes the "live-ready" line in the pitch).

**Ari**
- **You do:** pick the 3 best demo moments (ideally Gane's open hands flagged before a poke lands).

---

## Phase 7: Ship it

### 7.1 Repo polish
**Build**
- README: demo GIF at top, the problem in one line, how it works (simple diagram), results table with honest numbers, quickstart, limitations, roadmap, "not affiliated with Jabbr".
- Demo GIF from Ari's 3 picked moments with ffmpeg (a few seconds each, overlay on).
- Release trained weights via GitHub Releases. `labels/` documented so anyone can rebuild the dataset.
- Clean up CLI help, configs, and tests.
- Test the README quickstart on a fresh clone.

**Done when**
- A fresh clone + README quickstart runs `deepfoul detect` on a user-supplied video.

**Ari**
- **Claude chat:** draft the X thread, and write the Jabbr pitch email together (ask: paid trial project).
- **You do:**
  - Final read of the README and the numbers.
  - Post the thread and send the email from ari@ar13.dev.

---

## v2 backlog (do not build in v1)
- Knees/kicks to a grounded opponent (pose already shows hands/knees on the mat; rule definitions vary by commission)
- Strikes to the back of the head (needs head orientation)
- Fence grabs, glove/short grabs, headbutts
- Live camera input
- Re-ID model (e.g. torchreid) if color profiles aren't enough
- Multi-camera / 3D to solve the "2D overlap isn't contact" problem
