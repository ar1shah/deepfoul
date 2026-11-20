# DeepFoul Roadmap

## ▶ Now: 1.2 Detection and tracking

**Before the kickoff (you, left over from 1.1)**
1. Open `outputs/shots_check/` (two cut sheets + `report.md`). Each pair should be two different camera angles. I already checked all 20 by eye, so just note anything that looks off: fake cuts (same angle) or obvious missed cuts.
2. **Pick the golden set** (10 to 30 s each, live speed, train fights only, no replays). Times are from the original video, as VLC shows them. The preview clips have the original time burned in at the top left (`mm:ss.cc  f<frame>`), so you can read times straight off them.
   1. **Eye poke:** the poke previews in `outputs/golden_candidates/` (see `index.md`). Munhoz vs O'Malley (12:07) or Battle vs Loosa (08:09).
   2. **Low blow:** Holland vs Malott (05:02 or 05:52).
   3. **Clean stand-up exchange**, both fighters fully visible: O'Malley vs Vera or Magomedov vs Petrosyan.
   4. **Clinch against the cage:** Barber vs Grasso, or any train fight.
   5. **Ref stepping in** between the fighters (a break or a stoppage): any train fight.
   6. **(Bonus) women's stand-up exchange:** Barber vs Grasso.
   Don't use Edwards vs Muhammad or Garbrandt vs Long (video quality), or any val or test fight. Avoid between-round segments (dissolves aren't detected as cuts).
   Format: `1. ufc276_munhoz_omalley 11:55-12:20 eye poke`

**Kickoff prompt**
```
Read CLAUDE.md, ROADMAP.md and PROGRESS.md. Plan section 1.2 only. Wait for my approval before writing code.
Shot detection issues from 1.1: <list or "none">.
Golden set (write these to labels/golden.csv first):
<paste your golden list>
```

**Claude Code does**
1. Write the golden list to `labels/golden.csv` (converting mm:ss to seconds).
2. Person detection (rtmlib YOLOX) + ByteTrack within each shot. Read frames with `iter_fight_frames` and reset the tracker whenever `shot_index_at_frame` changes (shot files are already cached for all 9 train fights). Treat cuts as ±1 frame: some uploads blend one frame across a cut.
3. Filters: drop small boxes and boxes outside the cage area (crowd, camera crew). Thresholds in config.
4. Debug renders (boxes + track IDs + timestamp) for every golden clip in `outputs/renders/1.2/`.

**Done when**
- On the golden clips, the 2 fighters and the ref each keep a stable track ID through stand-up exchanges, and the crowd isn't tracked.

**You do**
1. Watch every debug render.
2. Write down each problem as `clip timestamp what happened` (e.g. `golden 2, 00:12, fighters swapped IDs in clinch`).
3. Paste the list into the 1.3 kickoff prompt.

---

## How every section works

Work is sequential: Claude Code finishes its part, then Ari does his. Nothing runs at the same time.

**The loop for every section:**
1. **Start a new Claude Code session** (plan mode) in `C:\Users\AriSh\Documents\projects\deepfoul` and paste the section's **Kickoff prompt**.
2. **Send the plan to Claude chat** (the DeepFoul project). Apply any tweaks, then approve.
3. **Claude Code builds and verifies** everything under **Claude Code does**, updates `PROGRESS.md`, commits and pushes. Some sections have a **pause point** where Claude Code stops mid-session and waits for a reply from Ari.
4. **Ari does the steps under You do**, in order.
5. If the section has a **Send to chat** item, send it before starting the next section.

**Who does what:**
- **Claude Code**: all code, installs, downloads, batch runs, renders, reports, edits to every CSV except `events.csv` and commits. If something can run from the repo, Claude Code does it.
- **Claude chat / Cowork**: web research (finding fights and URLs) and writing (posts, emails), plus reviewing plans and results.
- **Ari**: logins, approvals, watching footage, judgment calls and **all labeling**. Ground truth must come from a human, never from Claude.

**Rules of thumb for Ari:**
- **Never hand-edit CSVs** (except reviewing Cowork drafts). Give the information to Claude Code in the next kickoff prompt, or as a reply at a pause point, and it writes the file. The labeling app writes `events.csv`.
- **Watch footage in VLC** (or any player that shows exact times). When you note a moment, write it as `fight_id mm:ss` (e.g. `ufc276_omalley_munhoz 03:41`).
- **The standard kickoff prompt** is: `Read CLAUDE.md, ROADMAP.md and PROGRESS.md. Plan section X.Y only. Wait for my approval before writing code.` Sections add extra lines when needed.

---

## Footage guide

### Three kinds of footage
1. **Full fights**: the main source. Used for tracking, mining stoppages, negatives and most positives.
2. **Foul compilations**: YouTube "eye poke / low blow compilation" videos. Only used to top up positives (e.g. `comp_eyepokes_01`, `split=train`).
3. **Test fights**: Ciryl Gane's fights. Held out, never trained or tuned on.

Every video gets one row in `labels/fights.csv`. `deepfoul fetch` downloads every row's `source_url` to `data/raw/<fight_id>.mp4` and skips files that already exist.

### Quality bar for any video
- Full fight at **normal speed**, 720p minimum (1080p preferred)
- Official uploads first (the UFC's own YouTube channel posts free full fights)
- No reaction channels, commentary overlays, mirrored or cropped re-uploads
- Mostly UFC broadcasts from 2022 onward, so the cameras and graphics look consistent
- Footage stays in `data/raw/`, private, never redistributed

### Videos with no free official upload
Recent fights (like Gane vs Hokit at UFC 334) often aren't on YouTube for free yet. Options, in order:
1. Leave `source_url` blank and keep the row. Add the URL once the UFC posts it.
2. If you get a legit copy another way, save it as `data/raw/<fight_id>.mp4` yourself. `fetch` skips files that are already there, so it just uses yours.

Test fights are only needed in Phase 6, so a missing test video doesn't block anything before then.

### Starter list (15 to 20 full fights)
| Bucket | How many | What | Split |
|---|---|---|---|
| Test | 2 to 3 | Aspinall vs Gane (UFC 321), Gane vs Hokit (UFC 334), plus any other recent Gane fight with fouls | `test` |
| Foul fights | 8 to 10 | An eye poke or low blow visibly paused or ended the action. Roughly half eye pokes, half low blows. | 2 to 3 `val`, rest `train` |
| Clean fights | 5 to 7 | No notable fouls, mostly stand-up striking | 1 `val`, rest `train` |

Variety: at least 2 women's fights, at least 1 fight with similar or dark shorts, a mix of striking-heavy and grappling-heavy fights, a mix of weight classes.

### Growth targets (Phases 2 to 5)
- 30+ train fights, 5 to 6 val fights. The val fights must contain at least 8 real fouls between them, or the val metrics mean nothing.
- 150+ eye pokes, 150+ low blows, 200+ hard negatives.
- **Compilation leakage rule:** compilations often include famous pokes, including Gane's. Skip any compilation segment from a test fight while labeling.

---

---

## Phase 1: Perception pipeline (the "eyes")

1.2 is in progress (see **Now** at the top).

### 1.3 Identity: ref filter and red/blue

**Kickoff prompt**: standard, plus your tracking issue list from 1.2.

**Claude Code does**
1. **Shorts colors first:** for each fight in `fights.csv`, save one frame where both fighters and the broadcast name graphic are visible to `outputs/shorts_check/<fight_id>.jpg`. Read off the shorts colors and which fighter is in which corner, fill `red_shorts` / `blue_shorts`, and flag fights where the two colors are similar.
2. **Pause point:** stop and ask Ari to verify the colors.
3. Per-fight color profiles: sample torso and leg regions from early frames, build HSV histograms per person.
4. Ref detection: the person with a shirt + long pants profile, plus a fallback ("the third person who never fights").
5. Corner assignment from the shorts colors. Carry identity across shot cuts by matching profiles.
6. Re-render the golden clips with red/blue boxes and no box on the ref, in `outputs/renders/1.3/`.

**You do (at the pause point)**
1. Open each image in `outputs/shorts_check/` and compare it to what Claude Code wrote. A wrong corner flips every label for that fight.
2. Reply with `colors ok`, or with corrections like `ufc276_omalley_munhoz: red = purple, blue = black`.

**Done when**
- Golden clips render with correct red/blue labels, nothing on the ref, and identities survive the cuts in those clips.
- It works on the similar-shorts fight.

**You do (after the session)**
1. Watch the `outputs/renders/1.3/` renders. Note any corner flips or ref boxes.
2. Paste the issues into the 1.4 kickoff prompt.

### 1.4 Pose extraction and caching

**Kickoff prompt**: standard, plus your identity issues from 1.3.

**Claude Code does**
1. RTMW whole-body on each fighter's tracked box. Save per frame: corner, box, 133 keypoints, confidences.
2. `deepfoul extract <fight_id>` runs shots + tracking + identity + pose and writes `data/keypoints/<fight_id>.parquet`. Resumable if interrupted.
3. `deepfoul extract --all`: every downloaded fight that isn't extracted yet, writing `outputs/extract_report.md` (fight, status, time taken, frames, % frames with both fighters found).
4. Test it on one full 3-round fight.
5. Tell Ari how long the full batch should take on the 2080.

**Done when**
- One full fight extracts end to end, and the report format works.

**You do**
1. Before bed, open a terminal and run:
   ```
   cd C:\Users\AriSh\Documents\projects\deepfoul
   uv run deepfoul extract --all
   ```
   Leave the PC on. If it stops, run the same command again and it picks up where it left off.
2. In the morning, open `outputs/extract_report.md`. Note any failed fights, or any fight where both fighters were found in under ~70% of frames.
3. Paste those notes into the 1.5 kickoff prompt.

### 1.5 Overlay renderer v1 + benchmark

**Kickoff prompt**: standard, plus the extract report notes from 1.4.

**Claude Code does**
1. `deepfoul render <fight_id> --start --end`: red/blue boxes, corner labels, an optional skeleton toggle, and a timestamp.
2. `deepfoul bench`: processing speed (fps) for detection, pose and the full pipeline on the current GPU.
3. Render every golden clip with skeletons on, in `outputs/renders/1.5/`, and run the benchmark.
4. Look into any fights flagged in the extract report.

**Done when**
- The golden set renders cleanly with skeletons, and the benchmark numbers are in PROGRESS.md.

**You do**
1. Watch every golden render. On the poke clip, check that the fingertip and eye points follow the actual hand and eye.
2. Optional: post a clip or screenshot if you're building in public.

**Send to chat:** the benchmark numbers plus anything that still looks wrong in the renders.

---

## Phase 2: Data engine

### 2.1 Labeling app

**Kickoff prompt**: standard.

**Claude Code does**
1. A local FastAPI app + a single HTML page, started with `uv run deepfoul label`, which opens `http://localhost:8000`.
2. **Queue mode:** plays a clip around each candidate moment (default 4 s before, 4 s after) with speed control.
3. **Full-fight mode:** `uv run deepfoul label --fight <fight_id>` opens a whole fight with a scrubber, for adding events by hand (used for val and test fights).
4. Keyboard shortcuts:
   - `1` eye poke, `2` low blow, `3` open hand, `4` hard negative, `5` clean
   - `R` / `B` offender corner, `C` contact yes/no, `M` slow-motion
   - `[` / `]` set start / end, `←` / `→` scrub, `Space` play/pause, `S` skip, `Enter` save and next
5. Saves every row to `labels/events.csv` immediately, so nothing is lost on a crash.
6. A practice queue made from the golden clips so it can be tested before mining exists.

**Done when**
- Ari can label 30 moments in under 10 minutes, mostly with the keyboard.

**You do**
1. Run `uv run deepfoul label` and do the practice queue.
2. Write down everything slow or annoying (missing shortcuts, clip too short, video loads slowly).
3. If the list is long, paste it into a quick fix session before 2.2. Otherwise paste it into the 2.2 kickoff prompt.

### 2.2 Stoppage finder

**Kickoff prompt**: standard, plus any labeling app fixes.

**Claude Code does**
1. Scans cached keypoints for candidate foul moments, using signals that don't need a foul model:
   - The ref moves quickly between or toward the fighters
   - Both fighters' motion drops sharply (action paused)
   - A fighter's hand goes to his own eye or groin region and stays there
   - A fighter bends over at the waist
2. Scores each moment and builds a ranked queue per fight (`deepfoul mine <fight_id>`, `deepfoul mine --all`).
3. Runs `mine --all` on every extracted train fight.
4. Checks the known fouls from `labels/notes/starter_fights.md` and reports how many land in the top 20 per fight.

**Done when**
- Most known fouls appear in their fight's top 20 candidates.

**You do**
1. Run `uv run deepfoul label --queue mined` and label every candidate:
   - Real fouls: `1` eye poke or `2` low blow, plus the offender corner and contact.
   - Stoppages for other reasons (a cut, a ref warning, a slip, a glove touch): `5` clean, with a short note. These also work as hard negatives.
2. Don't label val or test fights from the queue. They get full-fight labeling later.

### 2.3 Dataset builder + ingest

**Kickoff prompt**: standard.

**Claude Code does**
1. `deepfoul build-dataset`: turns `events.csv` + cached keypoints into training windows (default 2 s, configurable stride). Positives come from labeled fouls; negatives are sampled from clean time, with extra weight on hard negatives.
2. Splits by fight from `fights.csv`, and refuses to put a test fight in train or val (unit tested).
3. Writes a summary report: counts per class per split.
4. `deepfoul ingest`: one command for new fights (merge a Cowork draft CSV if present, then fetch -> extract -> mine), writing an ingest report.
5. A hard-negative candidate queue: moments where a closed fist or kick lands near a foul zone with no stoppage afterward (`--queue hardneg`).

**Done when**
- `build-dataset` produces the dataset and the summary, and the split test passes.

**You do**
- Start the data rounds below.

### Data rounds (repeat through Phases 2 to 4)

Targets before Phase 5:

| Bucket | Target | Where it comes from |
|---|---|---|
| Eye pokes | 150+ | Mined from full fights first, then compilations |
| Low blows | 150+ | Same |
| Hard negatives | 200+ | `--queue hardneg`, plus clean stoppages from mining |
| Clean | Automatic | Sampled by the dataset builder |
| Full fights | 30+ train, 5 to 6 val | Footage guide rules |

**One round:**
1. **Cowork:** "Find 5 to 10 more UFC full fights with documented eye pokes or low blows (plus 1 to 2 compilations), following the Footage guide in ROADMAP.md. Write them to `labels/fights_draft.csv` with the same columns as fights.csv, plus a notes file."
2. **You:** review the draft like the starter list: foul visible, full fight, normal speed, official upload if possible. Note each fight's window (`fight_id mm:ss-mm:ss`) and every foul (time, type, offender, clear live or only in replay).
3. **Claude Code** (short session): `Run deepfoul ingest on labels/fights_draft.csv with these fight windows: <paste>. Show me the ingest report.` For a big batch, run `uv run deepfoul ingest` yourself overnight instead.
4. **You:** label the new queues:
   - `uv run deepfoul label --queue mined`
   - `uv run deepfoul label --queue hardneg`
   - For compilations: `M` on slow-motion clips, and skip any Gane segment (leakage rule).
5. **Claude Code** (same short session or the next one): `uv run deepfoul build-dataset`, then report the counts.

**Send to chat:** the dataset summary once you hit about 50 positives per foul, and again before Phase 5.

---

## Phase 3: Features, rules and evaluation

### 3.0 Label the val fights (you, before 3.1)
1. For each val fight: `uv run deepfoul label --fight <fight_id>`.
2. Watch the whole fight and mark every eye poke, low blow and open hand. An unlabeled foul turns into a fake "false flag" and drags every metric down.
3. Do it in one pass per fight, without any model output on screen.

### 3.1 Evaluation harness

**Kickoff prompt**: standard.

**Claude Code does**
1. Event matching (±1.0 s default). Per foul type: recall, precision and false flags per fight.
2. `deepfoul eval --split val`: prints a table and saves `outputs/eval/<run>/report.md`, listing every miss and false flag (fight + time).
3. Unit tests on made-up predictions and labels, plus dummy "always flag" and "never flag" detectors that give the expected numbers.

**Done when**
- The tests pass and the dummy detectors produce sane reports.

**You do**
- Nothing new. Keep doing data rounds if you have time.

### 3.2 Feature layer

**Kickoff prompt**: standard.

**Claude Code does**
1. Keypoint subset, normalization by torso length, confidence masking.
2. Cross-fighter geometry: fingertips relative to the opponent's eye region; feet, shins and knees relative to the opponent's groin zone.
3. Motion features: velocities, head jerk, torso bend, hand-to-own-face and hand-to-own-groin distance, finger extension score.
4. Feature plots for the golden poke and low blow clips in `outputs/features/`.

**Done when**
- The poke plot shows fingertip-to-eye distance dropping at the poke and the reaction features spiking after it.

**You do**
1. Open the plots next to the golden clip in VLC. Do the spikes line up with what you see?
2. If not, note the timestamps and put them in the 3.3 kickoff prompt.

### 3.3 Rule detectors v0

**Kickoff prompt**: standard, plus any plot issues from 3.2.

**Claude Code does**
1. `open_hand`: fingers extended, pointing at the opponent's face, within range.
2. `eye_poke`: a fingertip enters the eye zone, then a reaction within the lookahead (hand to face, head snap, action stops).
3. `low_blow`: a foot, shin or knee enters the groin zone, then a reaction (bending at the waist, hands to the groin, action stops).
4. Event logic: window scores -> events (hysteresis, merging nearby flags, cooldown, offender corner).
5. `deepfoul detect <fight_id>` writes `outputs/<fight_id>_events.json`.
6. Runs `eval --split val` and builds a **review reel**: one video of every miss and false flag, each captioned with fight, time, and predicted vs labeled, in `outputs/eval/<run>/review_reel.mp4`.

**Done when**
- The rules fire on the golden poke and low blow clips and stay quiet on the clean clip.
- The val eval report and the review reel exist. The numbers will be rough, and that's fine.

**You do**
1. Watch the review reel. Put every mistake into one bucket: tracking failure, keypoint jitter, camera angle, wrong threshold, or label mistake.
2. Count the buckets.

**Send to chat:** the eval table and the bucket counts. Together we pick what 4.1 fixes.

---

## Phase 4: Make it not suck

### 4.1 Error-driven fixes

**Kickoff prompt**
```
Read CLAUDE.md, ROADMAP.md and PROGRESS.md. Plan section 4.1 only. Wait for my approval before writing code.
Buckets from the 3.3 review: <counts>. Fix these first: <the 2 to 3 we picked>.
Label mistakes to fix: <fight_id mm:ss what's wrong>.
```

**Claude Code does**
1. Fix the label mistakes Ari listed.
2. Fix the chosen buckets (likely: keypoint smoothing, ID handling in clinches, depth ambiguity).
3. Tune thresholds on **val only**.
4. Re-run eval and the golden set, and make a new review reel.

**Done when**
- Val metrics improve on the targeted buckets with no golden set regressions.

**You do**
1. Watch the new reel and recount the buckets.
2. Keep doing data rounds, prioritizing hard negatives that fooled the rules.

### 4.2 Replays and broadcast junk

**Kickoff prompt**: standard.

**Claude Code does**
1. Finds 10 to 15 example replay transitions in the downloaded fights and saves them to `outputs/replay_check/`.
2. **Pause point:** Ari confirms they're real replays.
3. Detects replays (slow motion and/or the replay transition graphic) so they get tagged, not flagged. Skips between-round segments.
4. Re-runs eval on 3 full val fights.

**You do (at the pause point)**
- Look through `outputs/replay_check/` and reply `ok` or list the wrong ones.

**Done when**
- No double flags from replays on 3 full val fights.

---

## Phase 5: Learned classifier

Don't start until the data round targets are hit. Under about 100 positives per foul, the model will just memorize.

### 5.1 Baseline model

**Kickoff prompt**: standard, plus the latest dataset summary.

**Claude Code does**
1. LightGBM on window features from 3.2, with class weights for the imbalance.
2. Thresholds picked on val.
3. A comparison table (rules vs LightGBM) in PROGRESS.md.

**You do**
- Nothing new.

### 5.2 Temporal model

**Kickoff prompt**: standard.

**Claude Code does**
1. A small PyTorch model (1D CNN or GRU) on normalized two-fighter keypoint sequences. Trains in minutes on the 2080.
2. Augmentation: horizontal flip with left/right keypoint swap, plus small scale and jitter noise.
3. `deepfoul train`, saved weights + config.
4. A three-way comparison table (rules, LightGBM, temporal) on val.

**Send to chat:** the comparison table. Together we pick the default detector (one model, or a mix like open hand from rules + fouls from the model).

### 5.3 Hard-negative loop

**Kickoff prompt**: standard, plus which detector we picked.

**Claude Code does**
1. Exports the model's top-scoring false flags on train fights to `--queue modelneg`.
2. **Pause point:** Ari labels the queue.
3. Retrains and re-runs the comparison.

**You do (at the pause point)**
1. `uv run deepfoul label --queue modelneg`. Most of these will be `4` hard negative.
2. Reply `done labeling` in Claude Code.

**Send to chat:** the new comparison table, before Phase 6.

---

## Phase 6: The demo

### 6.0 Label the test fights (you, before 6.1)
1. Make sure every test fight is downloaded (see "Videos with no free official upload").
2. Tell Claude Code to run `deepfoul extract` on the test fights (a quick session).
3. For each test fight: `uv run deepfoul label --fight <fight_id>`. Mark every eye poke, low blow and open hand in one sitting, without looking at any model output.

### 6.1 Final test run

**Kickoff prompt**: standard.

**Claude Code does**
1. Runs the chosen detector on the test fights **once** and saves the report.
2. Writes the test metrics and every flagged event (with timestamps) to PROGRESS.md.
3. No tuning after this. Any later change has to be justified on val and noted.

**You do**
- Read the report. Don't ask for tweaks based on the test results.

### 6.2 Overlay v2 (broadcast style)

**Kickoff prompt**: standard.

**Claude Code does**
1. Red/blue boxes with head and torso sub-boxes per fighter.
2. Foul zone highlight: the eye region or groin zone glows when a hand or foot enters it.
3. On a flag: the box flashes and a banner appears (e.g. `EYE POKE · BLUE -> RED · 0.87 · REVIEW`).
4. Skeleton shown only for about 1 s around a flag.
5. Side panel: foul log (time, type, offender, confidence) and open hands per round.
6. Its own visual identity, with no Jabbr or DeepStrike branding.
7. Renders of the Gane test moments in `outputs/renders/6.2/`.

**You do**
1. Watch the renders and write design feedback (colors, panel layout, banner timing).
2. Do a quick fix session with that feedback, or send it to chat first if you're unsure.

### 6.3 Stats graphic + live-readiness check

**Kickoff prompt**: standard.

**Claude Code does**
1. A shareable per-fight graphic (no footage): open hands per round per fighter, flagged fouls, a timeline. Saved as PNG.
2. Streaming check: runs the detector frame by frame with only the allowed lookahead, confirms it matches batch mode, and records latency and fps.
3. Exports short clips of every flagged moment in the test fights to `outputs/demo_candidates/`.

**You do**
1. Watch `outputs/demo_candidates/` and pick the 3 best moments (ideally Gane's open hands flagged before a poke lands).
2. Put the picks in the 7.1 kickoff prompt.

---

## Phase 7: Ship it

### 7.1 Repo polish

**Kickoff prompt**
```
Read CLAUDE.md, ROADMAP.md and PROGRESS.md. Plan section 7.1 only. Wait for my approval before writing code.
Demo moments: <your 3 picks>.
```

**Claude Code does**
1. A demo GIF from the 3 picks with ffmpeg (a few seconds each, overlay on).
2. README: the GIF at the top, the problem in one line, how it works (a simple diagram), a results table with honest numbers, quickstart, limitations, roadmap, "not affiliated with Jabbr".
3. Trained weights released through GitHub Releases. `labels/` documented so anyone can rebuild the dataset.
4. CLI help, configs and tests cleaned up.
5. Tests the README quickstart on a fresh clone.

**Done when**
- A fresh clone + the quickstart runs `deepfoul detect` on a new video.

**You do**
1. Read the README start to finish and check the numbers.

**Send to chat:** the README + results. Together we write the X thread and the Jabbr pitch email (the ask: a paid trial project). Then you post the thread and send the email from ari@ar13.dev.

---

## Done

### 0.1 Repo and environment ✅
uv project, `deepfoul doctor` / `fetch` / `pose-check`, label CSV headers, machine setup on the RTX 2080 (SAC off, `UV_PYTHON_INSTALL_DIR`), attribution off, noreply git email. Details in PROGRESS.md.

### 0.2 Starter fights ✅
15 fights in `labels/fights.csv` with fight windows (train 9 / val 4 / test 2), all downloaded. Known fouls in `labels/notes/known_fouls.csv` and `labels/notes/starter_fights.md`. `deepfoul fetch-check` writes `outputs/fetch_check/report.md` plus contact sheets. After 0.2, Kape vs Silva and Garbrandt vs Long swapped splits (Garbrandt's video is cropped and likely frame-interpolated, so it's in train now). Val = Cortes Acosta vs Delija, Kape vs Silva, Gaethje vs Holloway, Garry vs Prates.

### 1.1 Video IO and shot detection ✅
Frame reader (`iter_frames`, frame-exact vs ffmpeg on the 50 and 60 fps videos), shot files for all 9 train fights (AdaptiveDetector tuned to 2.0 / 30), `extract --shots`, `shots-check`, `golden-candidates` (4 clips with burned timestamps), `frame-check`. Found: Garbrandt-Long is 25 fps padded with duplicate frames, not interpolated; dissolves between rounds aren't detected as cuts.

---

## v2 backlog (do not build in v1)
- Knees/kicks to a grounded opponent (pose already shows hands/knees on the mat; rule definitions vary by commission)
- Strikes to the back of the head (needs head orientation)
- Fence grabs, glove/short grabs, headbutts
- Live camera input
- Re-ID model (e.g. torchreid) if color profiles aren't enough
- Multi-camera / 3D to solve the "2D overlap isn't contact" problem
