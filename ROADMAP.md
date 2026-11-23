# DeepFoul Roadmap

## ▶ Now: 3.1 Evaluation harness

**Before the kickoff (you)**
1. **Label the two new train queues** (left over from 2.3). `git pull`, then copy `data/queues/hardneg.csv` and `data/queues/random.csv` from the PC to the Mac (`data/` doesn't sync, and the Mac has no keypoint cache to mine from).
   - `uv run deepfoul label --queue hardneg` (128 items): strikes near the face or groin with no stoppage after.
     - Most are `4` hard negative + `C` contact (and `R`/`B` for who threw it, optional).
     - A real foul is `1`/`2` as usual, and an extended hand at the face is `3` open hand.
     - A clinch or ground scramble with nothing near a foul zone is `5` clean.
   - `uv run deepfoul label --queue random` (73 items): plain random moments. Usually `5` clean.
     - `Shift+6` round break, `Shift+8` foul timeout, `0` replay.
     - These clips fix the clean bias (today all clean examples come from mined clips).
   - Commit and push `labels/events.csv`.
2. **3.0: label the val fights.** For each val fight: `uv run deepfoul label --fight <fight_id>`.
   - Val fights: Cortes Acosta-Delija, Kape-Silva, Gaethje-Holloway, Garry-Prates.
   - Watch the whole fight and mark every eye poke, low blow and open hand. An unlabeled foul turns into a fake "false flag" and drags every metric down.
   - Do each fight in one pass, without any model output on screen. Use `Shift+6` / `Shift+8` / `0` for breaks, timeouts and replays if you mark them at all (unmarked time is fine in full-fight mode).
   - Commit and push `labels/events.csv`.

**Kickoff prompt**
```
Read CLAUDE.md, ROADMAP.md and PROGRESS.md. Plan section 3.1 only. Wait for my approval before writing code.
Labeled: hardneg <n>, random <n>, val fights <list>. Anything odd while labeling: <list, or "none">.
```

**Claude Code does**
1. Re-run `deepfoul build-dataset` and report the new counts (val now has windows; random clips add unbiased clean).
2. Event matching (±1.0 s default). Per foul type: recall, precision and false flags per fight.
3. `deepfoul eval --split val`: prints a table and saves `outputs/eval/<run>/report.md`, listing every miss and false flag (fight + time).
4. Unit tests on made-up predictions and labels, plus dummy "always flag" and "never flag" detectors that give the expected numbers.

**Done when**
- The tests pass and the dummy detectors produce sane reports.

**You do**
- Nothing new. Keep doing data rounds if you have time.

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

Done (see the **Done** list).

---

## Phase 2: Data engine

2.1 to 2.3 are done (see the **Done** list).

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
   - **Open task (from 1.3):** at least one train fight where both fighters wear truly similar shorts (e.g. black vs navy, or both black with different trim). None of the 15 starter fights has one; 1.3 used the closest pair (Barber-Grasso, gray vs black) as a stand-in. Re-run the 1.3 identity check on it once it's in.
2. **You:** review the draft like the starter list: foul visible, full fight, normal speed, official upload if possible. Note each fight's window (`fight_id mm:ss-mm:ss`) and every foul (time, type, offender, clear live or only in replay).
3. **Claude Code** (short session): `Run deepfoul ingest on labels/fights_draft.csv with these fight windows: <paste>. Show me the ingest report.` For a big batch, run `uv run deepfoul ingest` yourself overnight instead.
4. **You:** label the new queues:
   - `uv run deepfoul label --queue mined`
   - `uv run deepfoul label --queue hardneg` (after `uv run deepfoul mine --hardneg`)
   - `uv run deepfoul label --queue random` (after `uv run deepfoul mine --random`): unbiased clean
   - For compilations: `M` on slow-motion clips, and skip any Gane segment (leakage rule).
   - Full fights: `0` on a broadcast replay (never saved as clean).
5. **Claude Code** (same short session or the next one): `uv run deepfoul build-dataset`, then report the counts.

**Send to chat:** the dataset summary once you hit about 50 positives per foul, and again before Phase 5.

---

## Phase 3: Features, rules and evaluation

3.0 (label the val fights) and 3.1 are in **Now** at the top.

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
2. Fix the chosen buckets (likely: keypoint smoothing, ID handling in clinches, depth ambiguity). **ID handling in clinches** includes golden_04, a known failure in `labels/golden_baseline.csv`: at 14:03.8 the detector gives a wide box for the pair (labelled BLUE) plus a fragment (labelled RED) instead of one `merged` box. Also: merged clinch frames have no per-fighter pose (clinch knees). Identity re-runs from the keypoint cache (`extract --from-cache`, ~2 min per fight), so no GPU re-run is needed.
   Evidence from 1.5 (`outputs/extract_check/`, details in PROGRESS):
   - **Sticky merges:** in long one-fighter close-ups (doctor checks, foul timeouts) a lone fighter stays `merged` for the whole shot. This is 10 of 24 sampled merged tiles in Cortes Acosta-Delija and 5 of 24 in Holland-Malott. A box-width gate was tried and dropped, because real clinch boxes are no wider than one fighter.
   - **Fragment rule:** a fighter standing behind the other can be swallowed as a "fragment".
   - **Ref confusion:** a fighter in black shorts seen from behind gets costed as the ref (Holland, Delija).
   - **Intended behavior:** when one fighter fully hides the other, the visible one keeps their own corner and the hidden one is just missing, with no merged box. Two strict-xfail tests in `tests/test_identity.py` encode this.
   - **Kick at contact:** the kicker's and the victim's boxes become one `merged` box (golden_02 05:02.7), so a low blow has no per-fighter corner on the contact frame. The per-box `raw_label` is still in the cache.
   - **One skeleton per merged box:** top-down pose gives a merged box one skeleton, so the other fighter's keypoints vanish at contact (golden_02 05:02.7, close exchanges in golden 3-6). Candidate fix: while merged, run pose on each fighter's last known box. Pull forward if 3.3 shows it.
3. Tune thresholds on **val only**.
4. Re-run eval and the golden set (`id-golden`, compared against `labels/golden_baseline.csv`), and make a new review reel. Update the baseline for anything fixed.

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
2. Tell Claude Code to run `deepfoul extract <test fight ids>` (a quick session; `--all` skips test fights on purpose).
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

### 1.2 Detection and tracking ✅
Golden set in `labels/golden.csv` (golden 5 shifted to 11:10-11:30 to stay in the fight window). rtmlib YOLOX with scores kept + ByteTrack from Roboflow `trackers` (supervision dropped it), reset at every cut with ±1-frame guard frames, loose crowd filters plus a relative-height rule. `deepfoul track-golden` writes renders, per-detection parquet and a report. Stand-up is stable; clinch merges, ground shots and ref re-entry IDs go to 1.3. Detection ~20 fps on the 2080.

### 1.3 Identity: ref filter and red/blue ✅
Shorts colors for all 15 fights (`shorts-check`), RTMW pose once per kept box in the tracking pass (batched), and red/blue/ref/merged labels from fight-wide appearance profiles (shorts + torso, clothed shins for the ref), smoothed within tracklets. `deepfoul id-golden` renders + report. Cuts, ref crossings and re-entries are solved; the golden_04 clinch is only partly right, and Barber-Grasso (grey vs black) stands in for a similar-shorts fight.

### 1.4 Pose extraction and caching ✅
`deepfoul extract` runs one in-order pass per fight (shots, detect, track, RTMW, color descriptors on CPU threads, then identity) into `data/keypoints/<fight_id>.parquet` (corner, box, 133 keypoints, histograms). It resumes by shot-aligned parts, caps the rate at 30 fps, keeps the PC awake and writes `outputs/extract_report.md`. `--all` skips test fights and runs train before val. Identity re-runs from cache with `--from-cache` (~2 min per fight). A small weak-corner guard clears the seated officials in golden_04. `labels/golden_baseline.csv` records golden_04's clinch as a known 4.1 failure. Barber-Grasso: 9.9 fps, 90% both or merged.

### 1.5 Overlay renderer v1 + benchmark ✅
`deepfoul render` draws boxes, corner tags and optional skeletons from the keypoint cache: no GPU, `--speed 0.25`, `--sheet merged|single` contact sheets. `render --golden` renders to `outputs/renders/1.5/`. `deepfoul bench`: full extract pass 10.2 fps on the RTX 2080 (3.1 GB VRAM). The extract report now flags b/m of visible < 82% and visible < 88%. The three low fights are long foul stoppages and close-ups; sticky merges and the other identity misses go to 4.1.

### 2.1 Labeling app ✅
`deepfoul label`: a local FastAPI app + one plain HTML/JS page that streams the raw video with Range requests (Chrome/Edge; VP9/AV1/Opus), with queue mode (`data/queues/<name>.csv`, clip `t ± 4 s`) and full-fight mode (`--fight`), keyboard labeling and every save fsynced to `events.csv`. Practice queue of 30 train moments writes to `outputs/label_practice/`. Seeks under 0.5 s on all codecs; the browser frame matches the original-file frame. Ari's practice run: 30 moments in ~13:00.

### 2.2 Stoppage finder ✅
`deepfoul mine` ranks candidate foul moments from the keypoint cache. The signal that works is a **guard drop**: both fighters' hands come away from their heads after 10 s with a guard up, at the queue time + 3 s. Train known fouls in their top 20: 7/8 (untuned 3/8), val 3/6 held out. `mined.csv` = 20 per train fight + an `extra` item for each missed known foul (181 items). Labeling app: 0.5 / 2 s steps clamped to the clip, `W` widens, quick clean-reason keys `6`-`9` and `Shift+6`/`Shift+7`, `0` = replay skip, the rate is kept across fights.

### 2.3 Dataset builder + ingest ✅
Label audit + `relabel` edit queue: Delete/Backspace, `Shift+8` foul timeout. 7 fouls nobody had listed were found (Magomedov-Petrosyan x4 is now a foul fight; 23 known fouls). `build-dataset` turns events + reviewed clips into an index of 2 s windows (1599 train: 32 eye poke / 89 low blow / 61 open hand / 100 hard negative / 1317 clean, clean only from mined clips), with a summary and an automated audit; test fights are refused. `ingest` runs draft merge -> fetch -> extract -> mine with a report. `mine --hardneg` (128 strikes near foul zones) and `mine --random` (73) are ready to label.

---

## v2 backlog (do not build in v1)
- Knees/kicks to a grounded opponent (pose already shows hands/knees on the mat; rule definitions vary by commission)
- Strikes to the back of the head (needs head orientation)
- Fence grabs, glove/short grabs, headbutts
- Live camera input
- Re-ID model (e.g. torchreid) if color profiles aren't enough
- Multi-camera / 3D to solve the "2D overlap isn't contact" problem
