# DeepFoul Progress

Claude Code updates this file at the end of every ROADMAP section. Newest entries at the top of the log.

## Current status
| Phase | Section | Status |
|---|---|---|
| 0 | 0.1 Repo and environment | Done (RTX 2080) |
| 0 | 0.2 Starter fights | Done (15/15 downloaded) |
| 1 | 1.1 Video IO and shot detection | Done (shots for 9 train fights) |
| 1 | 1.2 Detection and tracking | Done (stand-up stable; clinch/ground to 1.3) |
| 1 | 1.3 Identity: ref filter and red/blue | Done (stand-up and cuts correct; golden_04 clinch partly wrong, see log) |
| 1 | 1.4 Pose extraction and caching | Done (all 13 non-test fights cached, 11h15m overnight) |
| 1 | 1.5 Overlay renderer v1 + benchmark | Done (golden renders with skeletons, 10.2 fps full pass) |
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
- **Benchmark** (`deepfoul bench`, RTX 2080, Barber-Grasso 500-560 s, 1080p30, 2.54 boxes per frame, `outputs/bench/2026-11-21_rtx-2080.md`):

  | stage | speed |
  |---|---|
  | decode only | 222 fps |
  | detection (YOLOX-m) | 20.6 fps (48.6 ms/frame) |
  | pose (RTMW-x, batched) | 23.0 fps (43.5 ms/frame, 17.1 ms per box) |
  | color descriptors | 57 ms/frame of CPU, on 3 threads overlapping the GPU |
  | **full extract pass** | **10.2 fps** |
  | identity from rows | 313 fps |
  | overlay render + skeletons | 50 fps |

  - GPU memory: 3.1 GB of 8 GB, flat during the pass.
  - The 13 non-test fights (405k processed frames) take **~11 h** of extract (measured: 11h15m).
- Shot detection (CPU decode): ~190 fps at 1080p30, ~110 fps at 60 fps. All 9 train fights: ~38 min.
- Labeled positives: eye_poke - / low_blow - / open_hand -
- Fights: train 9 / val 4 / test 2 (8 foul, 5 clean, 2 test). Known fouls: 16 (`labels/notes/known_fouls.csv`)
- Val fights: Cortes Acosta-Delija, Kape-Silva, Gaethje-Holloway, Garry-Prates (6 known fouls). Kape-Silva and Garbrandt-Long swapped splits after 0.2, because Garbrandt-Long's video is cropped and likely frame-interpolated.
- Best val metrics: -
- Test metrics: -

## Open issues
- **Speed:** the full extract pass is 10.2 fps (GPU-bound), ~11 h for the 13 non-test fights. That's fine as an overnight run, so no speed work was done. Options if a bigger batch needs it: fp16, a lower NMS candidate cap, or pose only on likely fighters.
- **Clinch identity** (4.1 "ID handling in clinches"): a tied-up pair is labelled `merged` (no per-fighter pose). golden_04 is a **known failure in `labels/golden_baseline.csv`**: at 14:03.8 the detector gives a wide pair box (BLUE) plus a fragment (RED) instead of one merged box. The seated officials labelled RED in 1.3 (13:58.6, 14:15.5) are fixed by the 1.4 weak-corner guard.
- **Sticky merges and other 1.5 identity findings** (4.1, see the 1.5 log):
  - lone fighters stay `merged` through long stoppage close-ups
  - the fragment rule swallows a fighter standing behind the other
  - a fighter in black shorts seen from behind is costed as the ref
  - the kicker's and victim's boxes merge at low-blow contact
  - **For 2.2 / 3.2:** a `merged` box still has one person's pose, and its per-box `raw_label` (the corner before merging) is in the cache.
- **For 3.2 (motion features):** Garbrandt-Long still has duplicate frames after the 30 fps cap (a 25 fps source padded to 60, processed at stride 2). Velocity features must use real `t` deltas and handle repeated frames (zero motion on a repeat is not "stopped").
- **Profile lookahead:** identity for a frame in the first 60 s of a fight uses the profile learned from those 60 s (same as 1.3). Live, this would be a calibration delay or learning from pre-fight footage. Noted, not changed.
- openmmlab's model server throttles to ~100 KB/s per connection. On a new machine, pre-download the two model zips from rtmlib's Hugging Face mirror (see the 0.1 log entry).
- **6 unofficial uploads** (Aspinall-Gane, Battle-Loosa, Garbrandt-Long, Holland-Malott, Munhoz-O'Malley, Edwards-Muhammad). Accepted for now; ask Cowork for official replacements in a data round. They can vanish or be re-uploaded with a different cut, which would break every timestamp for that fight. If a URL is swapped, the fight's window and known fouls have to be re-timed.
- Edwards-Muhammad is 480p (under the 720p bar). Kept: its extract row is fine (82% both or merged, 85% of visible).
- Garbrandt-Long is a Russian broadcast at 1694x1080, 60 fps. 1.1 found it's a **25 fps source padded with duplicate frames** (58% of frames repeat the previous one), not interpolated. Motion features (3.2) must not read a zero velocity on a duplicate frame as "stopped". Munhoz-O'Malley is 50 fps with one-frame cross-blends at cuts.
- Shot detection misses **dissolves** (gradual transitions). They only showed up in between-round segments (Garbrandt 12:16-13:36, Holland 20:13-21:13), which 4.2 skips.
- Slow-motion replays can fire bursts of fake cuts (Battle-Loosa 09:40-09:43, 4 shots under 1 s). Replays are tagged in 4.2.
- Jandiroba-Dern has a sponsor overlay (bottom right) burned into the official UFC upload.
- No Gane-Hokit (UFC 334) row yet; add it once a video exists. No truly similar-shorts fight yet (Barber-Grasso grey vs black is the stand-in); a data-round task.

## Switching machines (later, not now)
All work is on the RTX 2080 PC for now. When Ari moves to the 4090 PC (or any new Windows PC):
1. Turn off Smart App Control if it's on (it blocks unsigned DLLs).
2. Set user env var `UV_PYTHON_INSTALL_DIR=C:\Users\AriSh\.uv-python`.
3. Install uv, ffmpeg, yt-dlp and gh via winget; `gh auth login`.
4. `git clone` / `git pull`, then `uv sync`.
5. Pre-download the rtmlib model zips from the Hugging Face mirror (see the 0.1 log entry).
6. `uv run deepfoul doctor` must show `CUDAExecutionProvider` active. If it fails, pin onnxruntime-gpu to its CUDA 12 build before anything else.
7. `deepfoul fetch` to rebuild `data/raw/` from `labels/fights.csv`.

## Phase 1 summary (perception, done 2026-11-21)
**Pipeline.** `deepfoul extract` makes one in-order pass per fight over its window:
1. shots (PySceneDetect)
2. YOLOX detection with loose crowd filters
3. ByteTrack, reset at every cut
4. RTMW 133-point pose, batched, on kept boxes only
5. color descriptors
6. identity from fight-wide color profiles learned in the first 60 s: `red | blue | ref | merged | none`, smoothed within tracklets, with merges using the past only

Everything lands in `data/keypoints/<fight_id>.parquet`, one row per box: corner, raw label, box, track, 133 keypoints + scores, histograms. Identity re-runs from that cache in ~2 min with no GPU. `deepfoul render` draws any span from the cache, and `deepfoul bench` times every stage.

**Data.** All 13 non-test fights are cached (9 train, 4 val; 405k processed frames, ~2 GB). Frame rate is capped at 30 fps.

**Speed (RTX 2080).** Full pass 10.2 fps, ~11 h for the 13 fights, 3.1 GB VRAM. Detection (20.6 fps) and pose (23 fps) share the GPU about equally.

**Quality.** Golden set:
- stand-up, cuts, ref crossings and re-entries are correct
- both-or-merged is 71-98% per clip
- golden_04's clinch is a known failure (4.1)

Per fight, b/m of visible is 78-96%, median 90.

**Known blind spots** (most owned by 4.1):
- **Clinches:** a tied-up pair is one `merged` box with one person's pose, so clinch knees and the contact frame of a kick have no per-fighter corner.
- **Sticky merges:** a lone fighter stays `merged` in long stoppage close-ups.
- **Fragment rule:** a fighter standing behind the other can be swallowed.
- **Black shorts vs ref:** a fighter in black shorts seen from behind can be costed as the ref.
- **Occluded fighter:** when the ref walks in front of a fighter, the pose in that fighter's box is the ref's.
- **Breaks:** dissolves and replays aren't handled (4.2).
- **Profile timing:** the identity profile needs the first 60 s of the fight.

**What Phase 2 builds on.**
- **2.1 labeling app:** reads only video + `fights.csv` / `events.csv`. The renderer can draw an optional overlay.
- **2.2 stoppage finder:** reads the keypoint cache. Ref position and motion come from `ref` rows, fighter motion and hand-to-own-eye/groin from fighter keypoints. A long stoppage shows as merged/close-up stretches with a lone fighter (Cortes Acosta: 58% of the window is the poke break). Use `raw_label` where `corner` is `merged`.
- **Velocity features (3.2):** must use real `t` deltas (Garbrandt-Long's duplicate frames).

## Log

### 2026-11-21 · Section 1.5 · Overlay renderer v1 + benchmark
**Built:**
- **`viz/overlay.py`:**
  - draws red / blue / purple `RED+BLUE` boxes with corner + surname tags, optional thin grey `none` and yellow `ref`, the `mm:ss.cc f<idx> shot N` timestamp and `CUT` on guard frames
  - skeletons via rtmlib's `draw_skeleton` (coco133). Points under `kpt_conf_min` (0.3) are never drawn, and skeletons go only on red / blue / merged boxes.
  - highlight dots: **fingertips** (hand offsets 4/8/12/16/20) + **eyes** (1, 2, face 59-70) for pokes, **hips / knees / ankles / feet** (11-22) for low blows
  - `read_range` reads only the requested frames of a whole-fight parquet (pyarrow filter)
  - only cached frames are written, so a stride-2 fight renders at 25 or 30 fps with every overlay from its own frame
- **`deepfoul render <fight_id> [--start] [--end] [--skeleton] [--highlight poke|low_blow|both|none] [--speed 0.25] [--show-ref] [--hide-none] [--out]`:**
  - Times can be `mm:ss(.cc)` or seconds (`io/fights.parse_time`).
  - With `--speed` other than 1, frames are written at fps x speed with no audio. `clip_writer` takes an exact fractional rate.
  - **`--sheet merged|single [--n] [--seed]`** writes a contact sheet of random merged or one-fighter frames to `outputs/extract_check/<fid>_<kind>.jpg`.
  - **`--golden [--tag]`** renders every golden clip with skeletons from the **full-fight caches** to `outputs/renders/<tag>/`, plus 0.25x versions of the foul clips, with an index at `renders.md`. id-golden's `report.md` stays separate.
  - `id-golden`'s renderer now calls the same code.
- **`deepfoul bench [--fight] [--start] [--seconds]`** (`bench.py`, config `bench:`). After a 3 s untimed warm-up it times:
  - decode only
  - the full `perceive_span` pass, with its own per-stage detection, pose and descriptor timers
  - identity on those rows
  - the renderer with skeletons
  - peak VRAM (nvidia-smi sampled every 0.5 s)

  It writes `outputs/bench/<date>_<gpu>.md` + `.json`, including a batch-time estimate over all cached fights.
- **Extract report flags** (config `extract.check`, replacing the hardcoded 70%):
  - **check** = b/m of visible under 82%
  - **low visible** = visible under 88%

  The report was regenerated for all 13 fights. The 1.4 run's times are kept in `outputs/extract_report_1.4.md`.
- **Tests:** 10 new tests in `tests/test_overlay.py`, plus 2 strict-xfail clinch tests in `test_identity.py` (108 passed, 2 xfailed). They cover the keypoint indices, box colors, the ref hidden by default, masked keypoints never drawn, no skeleton on the ref, the range read, sheet frame picks, `parse_time`, the check flags and the bench math.

**Investigation (extract report follow-ups):**
- **Cortes Acosta-Delija (val), 27% both / 67% both-or-merged.** This is mostly not a tracking failure:
  - **58% of its window is the eye-poke stoppage.** The poke is at 03:29. The doctor check, replay review and restart run to ~08:45, then the KO at ~09:17 = `action_end_s`, so the window is right. Inside the stoppage it's mostly close-ups of one fighter.
  - In the action frames: both 42%, both-or-merged **90%** (in line with the other fights). These heavyweights spend a lot of time clinched (48% of action frames merged).
- **Merged contact sheets** (`<fid>_merged_{narrow,wide}_before.jpg`): 12 merged tiles at under 1.2x a single fighter's width + 12 over, sorted by eye.

  | fight | real clinch / pair in one box | one fighter hides the other | lone fighter wrongly merged |
  |---|---|---|---|
  | Cortes Acosta-Delija | 13 | 1 | 10 |
  | Barber-Grasso | 21 | 3 | 0 |
  | Holland-Malott | 19 | 0 | 5 |

  **The wrong ones are almost all from stoppages.** 8 of the 10 in Cortes are lone-fighter close-ups during the poke break (doctor check, a 73 s close-up shot). Holland's 5 are its low-blow timeouts and between-round split screens. Three causes:
  - **Sticky merge:** while merged, the merged box stands in as the missing fighter's last box, so a lone fighter in a long shot stays merged.
  - **Fragment rule:** a fighter partly behind the other (Cortes 00:39) had a box 72% inside the front fighter's box and was dropped as a "fragment".
  - **Delija from behind costed as the ref** (Cortes 00:31).
- **Holland-Malott and Garbrandt-Long, low b/m of visible:**

  | fight | one-fighter frames (% of visible) | close-ups | other box labelled none/ref | no other box |
  |---|---|---|---|---|
  | Holland-Malott | 22% | 55% | 38% | 6% |
  | Garbrandt-Long | 20% | 69% | 28% | 3% |
  | Cortes Acosta-Delija | 19% | 74% | 25% | 1% |

  Holland's sheet (`<fid>_single.jpg`, 24 tiles): 13 timeouts or close-ups (two low blows, so two long timeouts), 4 Holland in black shorts costed as the ref when seen from behind, 3 ground pair boxes labelled BLUE (the golden_04 failure type). Garbrandt is mostly close-ups (low priority, as Ari said).
- **Merge fix tried and dropped.** I replaced the aspect gate, which never fired (`merge_aspect_min` 0.45 is below a lone fighter's 0.47-0.61), with "merged box at least 1.25x either fighter's last separate width". It regressed golden_04 (93 -> 74% b/m, 3 -> 16 jumps) and check_02/03 (2 -> 25, 7 -> 23 jumps). The measured ratios show why: real clinch boxes are about one fighter wide (median 1.04-1.07x in golden_04 and check_02), the same as the sticky close-ups (median 1.12x). A fighter's own box is already wide with arms out, and a clinched pair is compact.
  - Following the plan (ship only without regressions), the gate was reverted and the problem goes to 4.1 with this evidence.
  - Identity is unchanged, so there was no 13-fight re-run and no old/new report columns.
  - **Intended behavior** is written into the `apply_merges` docstring and 2 strict-xfail tests: when one fighter fully hides the other, the visible one keeps their corner and the hidden one is missing.

**Decisions:**
- **The check moves to b/m of visible,** because plain both-or-merged mixes tracking quality with how much of the window is breaks and close-ups.
  - 82% = median (90) - 8. It flags exactly the three fights looked at here: Holland 78, Garbrandt 80, Cortes 81.
  - "low visible" < 88% flags Garbrandt (87) and Cortes (83).
  - All three are now explained, so on new fights these bars mark outliers worth a sheet, not failures.
- **Golden renders come from the full-fight caches,** the data every later section uses. Their both-or-merged matches the baseline on every clip.
- **0.25x renders have no audio,** because slowed audio is useless and only confusing.
- **No speed work:** 10 fps keeps the batch as one overnight run.

**Verified:**
- `render` on Barber-Grasso 14:00-14:10 (300 frames, 10.0 s, with audio) and on Munhoz-O'Malley, a stride-2 50 fps fight (250 frames at 25 fps, 10.0 s, so the overlay is in sync). 0.25x: 120 frames over 16 s.
- **Golden renders** in `outputs/renders/1.5/`, checked by eye on frame grids:
  - **golden_01 (poke):** fingertip dots on Battle's extended glove at Loosa's face at 08:09.6, eye contours on both faces. **At 08:08.4 the ref walks in front of Battle and the skeleton in Battle's box is the ref's** (pose takes the person in front).
  - **golden_02 (low blow):** hip / knee / ankle / foot dots follow Malott's kicking leg through 05:02.4-05:03.1. At contact (05:02.73-05:02.90) the boxes merge into one `RED+BLUE` box, a 4.1 item for low-blow attribution.
- **Benchmark:** see Key numbers. The full pass is 10.2 fps (1.4 measured 9.9).
- `id-golden --from-cache --tag 1.5`: no regressions, golden_04 known fail. golden_03 has 2 jumps vs 1 in the baseline, within tolerance. Its fight profile was re-learned from the full-fight rows by `extract --all`.
- `uv run pytest`: 108 passed, 2 xfailed. `git status`: nothing under `data/` or `outputs/`.

**Golden set:** no regressions (identity unchanged). Baseline not updated.

**Known issues:**
- The 4.1 items above (sticky merges, fragment rule, black shorts vs ref, merge at kick contact, pose on an occluding ref).
- `ffmpeg -shortest` drops the last video frame of a render with audio (299 of 300). It doesn't matter for review.
- The bench reports VRAM as flat (3.1 GB) because onnxruntime's memory arena fills during the warm-up.

**Next:**
- Ari: watch `outputs/renders/1.5/` (see ROADMAP Now "Before the kickoff") and send the benchmark numbers + render notes to chat.
- Then section 2.1 (labeling app).

### 2026-11-20 · Section 1.4 · Pose extraction and caching
**Built:**
- **`perception/extract.py`, one in-order pass per fight.** It runs detect -> filter -> ByteTrack -> RTMW (on kept boxes) -> color descriptors from the frame already in memory (no re-decode), then identity from those rows.
  - **`data/keypoints/<fight_id>.parquet`** has one row per kept box:
    - box and track: `frame_idx, t, shot_id, guard, det_idx, x1..y2, score, track_id`
    - pose: `kpts` (133 x 2, flattened), `kpt_scores`
    - colors: `hist_shorts/torso/thigh/shins` (80 bins, null = unknown), `shin_skin, torso_skin, clothed, torso_len`
    - identity: `raw_label, corner` (`red|blue|ref|merged|none`), `cost, margin, fallback, cost_red/blue/ref`
  - **Metadata:** fps, stride, perception and identity config hashes, model file names + SHA-256, rtmlib / onnxruntime versions, profile counts and warnings, timing.
  - **`<fight_id>_frames.parquet`:** one row per processed frame (shot, guard, detections, kept boxes).
- **Resume:** parts of whole shots (~60 s each) go to `data/keypoints/<fight_id>.parts/`, each written atomically. The tracker resets at every cut, so a resumed run gives exactly the rows of an uninterrupted one (unit-tested at stride 1 and 2, and verified on a real hard kill).
- **`extract.target_fps: 30`:** stride 2 for Garbrandt-Long (60) and Munhoz-O'Malley (50), 1 for the rest.
  - The grid is counted from the window's first frame, so resumed parts stay on it.
  - `frame_idx` and `t` stay the original file's.
  - ByteTrack and smoothing use the effective fps.
- **Color descriptors run on 3 CPU threads** (`extract.describe_workers`), overlapping the GPU, with results taken in frame order. In-line they cost ~45 ms per frame and dropped the pass to 5-8 fps. Threaded, the pass is GPU-bound again.
- **`deepfoul extract <ids> | --split X | --all [--from-cache] [--force] [--rebuild-profile] [--dry-run] [--shots]`:**
  - Builds missing shot files first and skips fights that are up to date (config hashes).
  - If only the identity config changed, it re-runs identity from cache.
  - `--all` = every non-test fight with a video, **train first, then val**. Explicit ids still reach test fights (for 6.0).
  - Each fight is isolated: a failure is logged to `outputs/extract_logs/<fight_id>.log` and the batch moves on.
  - Keeps Windows awake while running (`SetThreadExecutionState`, released on exit, no power settings changed).
  - Prints one progress line per part (% done, fps, fight ETA and run ETA).
- **`outputs/extract_report.md`** is rewritten after every fight. Per fight: status, time, frames (stride), fps, both, both or merged, **visible** (frames with any fighter), **b/m of visible**, ref %, profile counts, warnings, size, and a **check** mark under 70% both-or-merged.
- **Weak-corner guard** (`identity.weak_corner_guard`, before `apply_merges`). A red/blue box becomes `none` when all of these hold:
  - its corner cost is > 0.3 and above its ref cost
  - it's shorter than 0.7x the tallest box in the frame
  - its tracklet never matched that corner under 0.25 earlier (past only)
- **`smooth_track` is now a sliding window**, O(n) instead of O(n²) per tracklet. It gives the same output, tested against the old version on 300 random sequences.
- **`id-golden` uses the same pass and record format**, so `--from-cache` needs no GPU and no decode. Renders go to `outputs/renders/<tag>/` (default `1.4`). The report has a **vs baseline** column.
- **`labels/golden_baseline.csv` (committed):** expected both / both-or-merged / jumps per golden clip and check, a status (`pass | known_fail`), the known issue and the section that owns the fix. A clip regresses if both-or-merged drops by more than 3 points or it gains more than 2 jumps. golden_04 is `known_fail`, owned by 4.1.
- **Tests:** 15 new (98 total). They cover the stride rule, parts on the grid covering the window, resume equal to one run (stride 1 and 2), the parquet round trip with unknown histograms, identity from cache equal to in memory, the guard (official cleared and the pair box becomes `merged`, a middling-cost fighter kept, a downed fighter with a confident past kept), sliding vs old smoothing, `--all` selection, baseline verdicts and keep-awake.

**Decisions:**
- **The profile is learned from the fight's own cached rows** (first 60 s, one frame per 0.5 s), not from a separate GPU sampling pass. It's the same boxes, filters and pose. Teaching counts moved slightly (Barber-Grasso 31/31/15 -> 30/30/14), and golden_06 got better with it (3 jumps -> 0).
- **The guard needed two conditions beyond "weak and ref-like":**
  - **Height:** a plain cost threshold also dropped real fighters in close-ups. golden_03 lost 6 points of both-or-merged, golden_01 4.
  - **Tracklet history:** without it the guard dropped **Petrosyan crumpled on the mat** after the knockdown in golden_05 (11:27.7-11:30). His shorts barely show, so he looks clothed (0.94) and ref-like for a moment, just like a seated official.
  - Still small (about 20 lines), so it shipped here and didn't go to 4.1.
- **30 fps cap and train-before-val** order, per Ari.
- **Threads only for descriptors**, because the measured pass dropped below 9.5 fps without them (per Ari's condition).

**Verified:**
- **Golden set, GPU pass on the new code with the guard off:** every clip matches 1.3's numbers exactly, including check_01, which now runs at stride 2.
- **Golden set with the guard (from cache) and renders** in `outputs/renders/1.4/` (+ `show_ref/`):

  | clip | 1.3 both / b-or-m / jumps | 1.4 | vs baseline |
  |---|---|---|---|
  | golden_01 | 68 / 71 / 4 | 68 / 71 / 4 | ok |
  | golden_02 | 82 / 84 / 0 | 81 / 84 / 0 | ok |
  | golden_03 | 87 / 98 / 1 | 87 / 98 / 1 | ok |
  | golden_04 | 24 / 89 / 10 | 11 / 93 / 3 | known fail (4.1) |
  | golden_05 | 91 / 96 / 0 | 91 / 96 / 0 | ok |
  | golden_06 | 88 / 96 / 3 | 90 / 97 / 0 | ok |
  | check_01 | 90 / 98 / 0 | 90 / 98 / 0 | ok |
  | check_02 | 60 / 98 / 3 | 60 / 98 / 2 | ok |
  | check_03 | 66 / 94 / 11 | 64 / 93 / 7 | ok |

  - **golden_04, checked by eye on the render:** 13:58.6, 14:12.1 and 14:15.5 are now one `RED+BLUE` box with the officials unlabelled. 14:03.8 is still wide BLUE + fragment RED (4.1). Its "both" drops because more clinch frames are correctly `merged`.
  - **golden_05:** Petrosyan stays BLUE when he goes down.
- **One full 3-round fight end to end:** `deepfoul extract ufc258_barber_grasso` (15.3 min window, 27,602 frames).
  - **Resume:** hard-killed after part 3 of 14 (taskkill, like a power cut); 3 parts were on disk. The same command resumed from part 4 and finished. `grid_warnings` is empty (every part started on its planned frame), and the frames are contiguous.
  - **Speed:** 9.9 fps over the run, 46 min total GPU time (8 + 38).
  - **Result:** 79,598 boxes, 131 MB. Both 45%, both or merged 90%, 92% of frames with a fighter, ref 72%. Profile 30/30/14, no warnings.
  - **Matches the golden caches:** for the golden_04 and golden_06 spans, 100% of boxes are identical, the max keypoint difference is 0.0 px, and corners agree on 98% (a different profile, and tracklets that start before the clip).
  - **Identity-only re-run:** `extract ufc258_barber_grasso --from-cache` takes 1m47s, no GPU.
- `extract --all --dry-run` lists 9 train fights, then 4 val, no test fights, Barber-Grasso up to date.
- `uv run pytest`: 98 passed. `git status`: nothing under `data/` or `outputs/`.

**Batch estimate (RTX 2080, measured 9.9 fps):**
- **Left to run:** ~377k frames (train ~242k, val ~135k), so **~10.6 h of GPU**.
- **Overhead:** about 2-3 min of identity per fight, plus shot detection for the 4 val fights (~30 min, CPU).
- **Timeline:** about **11.5 h** in total, with **train done after about 7 h**. It's resumable, so a run that's still going in the morning can be left to finish or restarted later.

**About the 70% "check" bar:** Barber-Grasso, a grappling-heavy fight, gets 90% both-or-merged. "Both" alone is only 45%, because clinches are `merged`, so the bar must stay on both-or-merged. 98% of its frames have a fighter, so the between-round breaks don't move it much. Recommendation: keep 70% for the morning check, read it next to **b/m of visible** (a fight low on the first but high on the second is crowd and close-up shots, not tracking failure), and re-set it after the batch from the 12-fight spread.

**Golden set:** no regressions. golden_04 is recorded as a known failure (4.1) and is better than 1.3 (10 -> 3 jumps). golden_06 and check_03 improved.

**Known issues:**
- **golden_04 at 14:03.8:** wide pair box BLUE + fragment RED (4.1).
- **Merged clinch frames have no per-fighter pose** (clinch knees, 4.1).
- **Speed numbers in `outputs/renders/1.4/report.md` are stale** for clips last cached before threading (golden_01/03/05/06 show 5.7-7.1 fps). Barber-Grasso is the real number.
- Model load prints onnxruntime "nodes not assigned to the preferred provider" warnings (shape ops on CPU, harmless).

**Next:**
- Ari: run `uv run deepfoul extract --all` overnight (see ROADMAP 1.4 "You do"). In the morning, read `outputs/extract_report.md` and note failed fights and any **check** marks for the 1.5 kickoff.
- Then section 1.5 (overlay renderer v1 + benchmark).

### 2026-11-20 · Section 1.3 · Identity: ref filter and red/blue
**Built:**
- **Shorts colors** (step 1, confirmed by Ari):
  - `deepfoul shorts-check` writes contact sheets from 3 min before `action_start_s` to 1 min after, and `--save fight_id=mm:ss` saves the chosen frame to `outputs/shorts_check/<fight_id>.jpg`. It also writes `outputs/shorts_check/report.md`.
  - `red_shorts` / `blue_shorts` are filled for all 15 fights, mostly from the broadcast trunk-color graphic. Format: main color first, accents after `+`, validated against `identity.colors`.
- **`perception/pose.py`** (`PoseEstimator`): rtmlib RTMPose with the RTMW model `Wholebody` uses for the profile's mode. All of a frame's kept boxes go into **one ONNX call**, because the model's batch dimension is dynamic. Keypoints are identical to rtmlib's per-box loop (max diff 0.0 px), and it's 46 ms vs 70 ms per frame.
- **`track_stream(..., pose=...)`:** RTMW runs in the same in-order pass as detection and tracking, **only on kept boxes** (guard frames included). Filtered boxes never get pose.
- **`perception/appearance.py`:** pose-guided regions drawn as strokes scaled by torso length:
  - shorts = hips to 35% of the way to the knees
  - torso quad
  - lower thigh
  - shins
  - upper cheeks, as the skin reference
  - HSV histograms are 18 hue x 4 saturation chromatic bins plus 8 value bins for dark and grey pixels. A region with under 40 visible pixels is unknown, never guessed.
  - **clothed** score = mean of "shins aren't the person's own face skin" and "shins look like the fabric at the hips".
- **`perception/identity.py`:**
  - **Fight profile:** seeded from the color names, then learned on the first 60 s of the fight window, every 0.5 s, in two rounds. A frame teaches the profile only when, among the big boxes (≥ 0.45x the tallest), one clearly matches red, one clearly matches blue, the two don't overlap and neither has clothed shins. It learns red/blue shorts, red/blue torso (bare skin or a top) and ref clothing.
  - **Drift / swap warnings** go to the report.
  - **Per-frame assignment:** each of {red, blue, ref} takes one box or stays empty, solved exhaustively (at most 7^3 combos). Unassigned boxes cost a flat 0.7.
  - **Fighter cost:** shorts and torso similarity, each weighted by how well that cue separates the two fighters (tops in women's fights, shorts in men's). Penalties for clothed shins, and for a torso that looks like the ref's shirt more than this fighter's torso.
  - **Ref cost:** clothed shins plus the ref clothing profile. **Third-person fallback:** with exactly 3 people and both fighters found, the third is the ref.
  - **Smoothing:** majority vote within a ByteTrack tracklet over ±0.5 s, and a tracklet is split when a new label holds for 8 frames (an ID swap). After that, labels are de-duplicated per frame.
  - **`apply_merges`** (past frames only) marks a box `merged` when the other fighter's box disappeared, or became a fragment inside it, and the other fighter's last box (≤ 0.5 s ago, same shot) overlapped it. In a merged frame no other box can be red or blue.
  - **Size gate:** red/blue boxes shorter than 0.35x the tallest box in the frame become `none`.
  - `check_lookahead` refuses settings that need more future than `lookahead_s`.
- **`perception/id_clip.py`** has four passes:
  - GPU: detect, track and pose, cached to `data/cache/identity/<clip>_{tracks.parquet,pose.npz,timing.json}`.
  - CPU: re-decode and compute descriptors.
  - Identity, which uses no pixels.
  - Render: red/blue boxes `RED Holland`, purple `RED+BLUE`, thin grey `none`, and nothing on the ref. With `--show-ref` the ref gets a thin yellow box labelled `ref`.
- **`deepfoul id-golden [--only] [--from-cache] [--no-render] [--show-ref] [--rebuild-profiles] [--no-checks]`:**
  - writes renders to `outputs/renders/1.3/` (`show_ref/` for the ref versions), profiles to `data/cache/identity/<fight>_profile.npz`, and `outputs/renders/1.3/report.md` (`viz/id_report.py`).
  - the report has per-clip and per-shot stats (both found, corner-label jumps, ref frames, possible missed ref, merged), Ari's 1.2 issues #2 to #8 with what happens now at each time (`identity_issue_checks` in the config), profile teaching counts and warnings, and speed.
- **Tests:** 17 new identity tests, 2 new fights tests (83 total). They cover histograms (black vs navy, the low-saturation path), seed coverage, clothed shins, a fighter in black shorts vs a ref in black pants, a dressed ref in close-up, assignment with `none`, the third-person fallback, profile learning and its warnings, drift and swap warnings, the tracklet split vs a one-frame blip, the lookahead check, de-duplication and three merge cases (closing into one box, walking away, a fragment inside).

**Decisions:**
- **No IDs are carried across frames.** Labels come from fight-wide profiles, so cuts and ref re-entries (issues #3, #7, #8) need no special handling.
- **Profiles are learned from the start of the fight only**, so the method stays live-compatible. The color names are the seed, and they're also what the drift/swap check compares against.
- **Black seed range:** it originally overlapped every hue at low brightness, so "black" matched red. It now covers only pixels below `v_min`, or dark low-saturation ones. White starts at V 150, because white shorts read grey in shadow (Loosa).
- **Clothed shins:** the first version compared shins to the lower thigh, which is also true for a bare-legged fighter. It now compares them to the shorts region (one fabric from hip to ankle means pants).
- **Merge detection by time, not color:** backprojecting the missing fighter's colors didn't separate real clinches from a fighter alone, because dark background matches black shorts. The time-based rule only looks at the past.
- **Adaptive cue weights:** in Barber-Grasso the tops (black vs white) separate the fighters better than the shorts, and fixed weights caused swaps in close-ups with the shorts cut off.
- **Ref torso penalty:** in men's fights both torsos are bare skin, so the torso weight is small, and a ref in close-up with his legs out of frame took Battle's label. The penalty fixed that.
- **The ref is never drawn** in the main renders. `--show-ref` exists so Ari can tell a hidden ref from a missed one.

**Verified:**
- `deepfoul id-golden --rebuild-profiles`: a full GPU run on 6 golden clips and 3 checks, then the `--show-ref` renders from cache. Detector and pose both on `CUDAExecutionProvider`, pose batched.

  | clip | both fighters | both or merged | corner jumps | notes |
  |---|---|---|---|---|
  | golden_01 Battle-Loosa | 68% | 71% | 4 | Correct through the ref crossing (#2). The ref close-up at 08:16 to 08:19 gets no fighter box; Battle at the frame edge is RED. Shot 38 is mostly a close-up with one fighter out of frame. |
  | golden_02 Holland-Malott | 82% | 84% | 0 | Correct at the 04:58 cut (#3). Black-shorts Holland is RED, the ref in black pants is not. |
  | golden_03 O'Malley-Vera | 87% | 98% | 1 | No swap at 08:53 or 09:09 (#4). `merged` only where both fighters share one box. |
  | golden_04 Barber-Grasso clinch | 24% | 89% | 10 | **Not fully correct, see Known issues.** |
  | golden_05 Magomedov-Petrosyan | 91% | 96% | 0 | Correct when the ref walks behind Shara (#7). |
  | golden_06 Barber-Grasso stand-up | 88% | 96% | 3 | Correct through the 08:41 and 08:48 angle changes (#8). All 12 sampled frames correct. |
  | check_01 Munhoz-O'Malley | 90% | 98% | 0 | Clean. |
  | check_02/03 Jandiroba-Dern (ground) | 60% / 66% | 98% / 94% | 3 / 11 | Tangled pairs are `merged`, separated fighters are correct. |

  Checked by eye on contact sheets of every render, at each of Ari's issue times plus 8 to 12 frames per clip.
- **Ref hidden in the women's fight:** golden_04 has the ref in 553/600 frames (3 possible misses), golden_06 410/600 (0). In the `--show-ref` renders the yellow box follows the ref, and the main renders draw nothing on him.
- **Profiles:** 22 to 77 teaching frames per fight, no drift or swap warnings.
- **Speed (RTX 2080, 7199 frames):**
  - detection 22.3 fps, pose 22.8 fps (2.8 kept boxes per frame)
  - **whole GPU pass incl. decode and tracking: 10.5 fps**
  - The CPU identity pass (re-decode + descriptors) adds roughly another minute per 20 s clip; it isn't optimized yet.
  - **For 1.4:** a 15-min fight at 30 fps (27,000 frames) needs about **43 min** for detect + track + pose, and ~86 min at 60 fps (Garbrandt-Long). The 9 train fights' windows add up to about 3.7 h of video, so roughly **6 to 7 h on the GPU**, plus the identity pass.
- `uv run pytest`: 83 passed. `git status`: nothing under `data/` or `outputs/`.

**Golden set:** tracking itself is unchanged from 1.2 (same detector, filters and ByteTrack). Identity is new; golden_04 is the one clip that doesn't fully pass.

**Known issues:**
- **golden_04 clinch (Barber-Grasso, 13:56 to 14:16) is not fully correct.** Pose on a box holding two people sees one of them. At times the detector gives a wide box for the pair plus a small box on one fighter: the wide box is labelled BLUE instead of `merged` and the fragment RED (13:58.6, 14:03.8, 14:12.1). Seated cageside officials sometimes get RED (13:58.6, 14:15.5), because their grey/dark clothes look like Barber's grey shorts and their size is close to a crouched fighter's.
- **Merged clinch frames are a blind spot** for clinch low blows (knees): there is no per-fighter pose there. Candidate for 4.1 "ID handling in clinches".
- Crowd members behind the fence can still get a corner label for a few frames in ground shots (Jandiroba-Dern), when the fighters are tangled and the slot is free.
- **"both fighters" is low by design** in close-ups with one fighter out of frame (golden_01 shot 38) and in clinches (`merged` instead).
- Still no truly similar-shorts fight: Barber-Grasso (grey vs black) is the stand-in, and it passes in stand-up (golden_06). "Find a similar-shorts fight" is a data-round task in ROADMAP.
- The identity CPU pass re-decodes the video. 1.4's single extract pass should compute descriptors from the frame already in memory.

**Next:**
- Ari: watch `outputs/renders/1.3/` (and `show_ref/` to tell a hidden ref from a missed one) and note corner flips or ref boxes as `clip timestamp what happened`. Decide whether the golden_04 clinch errors go to 4.1 or need more work before 1.4.
- Then section 1.4 (pose extraction and caching).

### 2026-11-19 · Section 1.2 · Detection and tracking
**Built:**
- `labels/golden.csv`: Ari's 6 golden clips (in seconds). Golden 5 was shifted to 11:10-11:30 (670-690 s) because 11:17-11:37 ran past Magomedov-Petrosyan's `action_end_s` (690); the fight window was left alone. `tests/test_golden.py` checks the schema, train split, length and that each clip is inside its fight window.
- `src/deepfoul/perception/detect.py`: `PersonDetector`, rtmlib's YOLOX (same model file as `Wholebody`, from the profile's `pose_mode`) via rtmlib's own `preprocess` + `inference`, then `parse_dets` reads the model's `[1, N, 5]` output and **keeps the scores**.
- `src/deepfoul/perception/filters.py`: `drop_reasons` with 5 loose rules (`small`, `rel_small`, `high`, `edge`, `max_people`), thresholds in `tracking.filters`.
- `src/deepfoul/perception/track.py`: `track_stream` (detect -> filter -> ByteTrack per frame, in order) and `track_clip`. The tracker resets on every shot change; frames `cut-1` and `cut` are guard frames (detected and drawn, not fed to the tracker). Track ids are `<shot_id>-<local_id>`.
- `src/deepfoul/viz/debug_render.py` (supervision box/label annotators, unconfirmed boxes white, dropped boxes grey with the rule name, `mm:ss.cc f<idx> shot N` and `CUT` on guard frames) and `viz/track_report.py`.
- `io/previews.py`: the ffmpeg pipe is now a shared `clip_writer` (used by golden previews and debug renders).
- CLI `deepfoul track-golden [--only id] [--no-render] [--no-checks]`: renders to `outputs/renders/1.2/`, every detection to `data/cache/golden/<id>_tracks.parquet` (with `kept`, `drop_reason`, `guard`; `track_id` empty for dropped/unconfirmed boxes, for 1.3's ref filter tuning), and `outputs/renders/1.2/report.md`.
- Config: `det.score_min`, `tracking.{cut_guard_frames, bytetrack, filters, checks}`. 3 extra check spans: Munhoz-O'Malley 04:10-04:40, Jandiroba-Dern 02:30-03:00 and 08:40-09:10 (the only non-golden train fights allowed).
- 7 new tests (64 total): detector parsing, each filter rule, guard frames, and the reset/guard/id logic with a fake detector and the real ByteTrack.

**Decisions:**
- **ByteTrack from Roboflow `trackers` 2.7.0** (Apache-2.0): supervision 0.32 no longer ships ByteTrack. Checked before coding: no torch in its base install (`uv tree`), license in its METADATA, API read from the installed source (`update` returns boxes in a different order with `tracker_id=-1` for unconfirmed tracks, so a `det_idx` column maps them back; `reset()` restarts ids from 0). CLAUDE.md stack table updated. Supervision stays for drawing.
- **rtmlib score workaround:** `YOLOX.__call__` returns boxes only and drops scores under 0.3. The bundled YOLOX ONNX has NMS inside, so we read its `dets` ourselves with `score_min` 0.1 (ByteTrack's second pass uses low-score boxes). Parsed boxes are identical to rtmlib's on a test frame.
- **Relative-height rule added** (`rel_small`: shorter than 0.4x the tallest box scoring >= 0.5). Without it, tight broadcast shots tracked background crowd (golden_03: 38 tracks in 4 shots -> 14). Absolute rules stay loose (`min_height_frac` 0.10, `min_bottom_frac` 0.35, `max_people` 6) per Ari.
- ByteTrack in fixed-rate mode (frames are fed strictly in order): `lost_track_buffer` 30, activation 0.5, high-conf 0.5, min IoU 0.1, 2 frames to confirm.
- **Lookahead note:** guard frame `cut-1` needs to know the next frame is a cut, i.e. 1 frame of lookahead. That's inside the 2.0 s rule; a live version needs a 1-frame delay or must drop that guard.
- The humanart YOLOX boxes are ~20% larger than the person (centred on them). It's the model, not our parsing (rtmlib gives the same boxes). Consistent, so fine for tracking; the "high" rule uses the box bottom, which sits a bit below the feet.

**Verified:**
- `deepfoul doctor`: CUDA still active after adding `trackers`; detector session on `CUDAExecutionProvider`.
- `deepfoul track-golden`: 6 renders + 3 checks + report. Contact sheets (6-12 frames each) of all 6 golden renders and all 3 checks, checked by eye:

  | Clip | Result |
  |---|---|
  | golden_01 Battle-Loosa (poke) | Both fighters keep one ID per shot (336/326 of 337 frames). The ref walks in front of the camera and gets a new ID per re-entry. |
  | golden_02 Holland-Malott (low blow) | 3 full tracks (2 fighters + ref) in every shot. |
  | golden_03 O'Malley-Vera (stand-up) | Fighters full in every shot; extra tracks are the ref entering. No crowd. |
  | golden_04 Barber-Grasso (clinch) | Fighters often detected as one merged box: 10-12 tracks per shot. Ref tracked. Cageside judges dropped. |
  | golden_05 Magomedov-Petrosyan (ref steps in) | Fighters full through the shot (558/555 of 572 frames), ref tracked when he steps in. |
  | golden_06 Barber-Grasso (women's stand-up) | Fighters full in every shot, ref tracked. |
  | check_01 Munhoz-O'Malley | Fighters full in all 5 shots. The ref is the third track wherever he's in frame (shots 11, 13-15), full in shot 15. |
  | check_02/03 Jandiroba-Dern | Stand-up shots: 3 full tracks (17.5 s shot with 523/522/520 frames). Ground shots fragment, and seated cageside officials get tracked. |

  Boxes dropped only by `high`: 0 in every golden clip, 1 + 49 in the checks, all low-score crowd above the cage line in ground shots.
- `uv run pytest`: 64 passed. `git status`: nothing under `data/` or `outputs/`.

**Golden set:** baseline established.

**Known issues:**
- Clinch merges, ground shots and ref re-entry IDs (see Open issues). Not in the 1.2 Done criterion (stand-up), handed to 1.3.
- Detection ~20 fps (see Open issues).
- supervision prints `template_mgs` FutureWarnings on import. Harmless.

**Next:**
- Ari: watch every render in `outputs/renders/1.2/` (`report.md` links them) and write problems as `clip timestamp what happened` for the 1.3 kickoff.
- Then section 1.3 (identity: ref filter and red/blue).

### 2026-11-19 · Section 1.1 · Video IO and shot detection
**Built:**
- `src/deepfoul/io/video.py`:
  - `iter_frames(path, start_s, end_s, stride)` yields `Frame(idx, t, image)` in order. It covers half-open `[start, end)` and seeks 2 s early, then decodes forward to the exact first frame at or after `start`.
  - `t` is the original-file time from the container pts, and `idx` is the frame index in the original file.
  - `iter_fight_frames(row, ...)` defaults to, and clamps to, the fight window.
  - `video_info` reports `start_time`, `r_frame_rate` and `avg_frame_rate`, and flags a nonzero start or VFR.
  - `frame_check` compares the reader against ffmpeg.
- `src/deepfoul/io/shots.py`: PySceneDetect AdaptiveDetector over the fight window, cached to `data/cache/<fight_id>_shots.json`.
  - Shots are **half-open** `[start_frame, end_frame)`, so a frame exactly at a cut belongs to the new shot.
  - `min_scene_len_s` is converted with each video's own fps.
  - `shot_index_at_frame` / `shot_index_at` is the hook 1.2 uses to reset tracks when the shot id changes.
  - Optional `shots.per_fight` overrides (currently empty) and a `content` detector option.
- `src/deepfoul/io/previews.py`: cut contact sheets, the shots report, and golden preview clips.
- CLI:
  - `deepfoul extract <ids...> | --split train --shots [--force]`: if the cached detector params, scenedetect version or fight window no longer match, it re-runs automatically and prints why.
  - `deepfoul shots-check <ids...> [--n 20]`
  - `deepfoul golden-candidates`
  - `deepfoul frame-check <id> [--t ...]`
- 12 new tests (55 total), on a synthetic video with known cuts.

**Decisions:**
- **Detector params tuned:** `adaptive_threshold` 3.0 → 2.0 and `min_content_val` 15 → 30, checked by eye on 5 fights:

  | Span | Defaults | Tuned |
  |---|---|---|
  | Garbrandt 400-520 s | 35 cuts, 25 fake | 10 cuts, all real |
  | Battle-Loosa 240-340 + 470-520 s | 5 real missed, 8 fake | 0 missed, 0 fake |
  | Munhoz 140-330 s | 3 missed | 0 missed, 0 fake |
  | Magomedov 10-190 s | 1 missed | 0 missed |
  | Holland 300-400 s | 2 missed | 0 missed |

  - Real cuts have a mean pixel diff (before → after) of 31 to 170. Fakes have 2 to 16.
  - Garbrandt's fakes came from its duplicate frames.
  - Battle-Loosa's fakes came from 1.33 s compression pulses in the re-encode, and its real cuts were missed because the noisy upload has a high baseline frame diff.
  - One global setting fixed both, so no per-fight override was needed. ContentDetector was tried for Garbrandt and dropped.
- Shot detection decodes from frame 0 instead of seeking, so PySceneDetect's frame numbers match the file exactly. Cuts before the window start are dropped.
- The frame reader uses OpenCV (pts time) and checks itself against ffmpeg. PyAV is installed but wasn't needed.
- Golden candidates skip Garbrandt-Long and Edwards-Muhammad (video quality), which leaves 4 clips. Clips cover ±10 s with the original audio, and the original time and frame index are burned in (`12:07.48  f36274`).
- Integer times in PySceneDetect mean *frames*, not seconds. All calls pass floats.

**Verified:**
- **Frame timing vs ffmpeg** (`deepfoul frame-check`), 5 off-grid times per video:
  - Munhoz-O'Malley (50 fps): mean abs pixel diff vs ffmpeg 0.00 at all 5. Neighbouring frames differ by 1.6 to 15.8, so the check really separates frames. Seek-then-decode equals decode-from-10-s-earlier (diff 0.00, same `t`).
  - Garbrandt-Long (60 fps): diff 0.00 at all 5, and seek equals linear decode. Because of the duplicate frames, ±1 frame is sometimes identical, so at those times only "not 2 frames off" is provable. At 709 s and 1311 s, +1 is clearly different (8.0, 11.9).
  - `video_info`: both have `start_time` 0, `r_frame_rate` = `avg_frame_rate`, and no flags.
- **Shots:** files for all 9 train fights:

  | Fight | Shots | Median |
  |---|---|---|
  | Garbrandt-Long | 150 | 6.0 s |
  | Holland-Malott | 193 | 5.5 s |
  | Battle-Loosa | 49 | 9.3 s |
  | Munhoz-O'Malley | 60 | 9.0 s |
  | Edwards-Muhammad | 42 | 8.6 s |
  | Magomedov-Petrosyan | 66 | 7.1 s |
  | Jandiroba-Dern | 204 | 5.6 s |
  | Barber-Grasso | 94 | 7.3 s |
  | O'Malley-Vera | 182 | 6.4 s |

- **Contact sheets** (Munhoz-O'Malley, Magomedov-Petrosyan, 10 cuts each): I checked all 20 by eye. All are real angle changes or replay wipes.
- **Golden candidates:** 4 clips (Holland 05:02 and 05:52, Battle 08:09, Munhoz 12:07), 20 s each, H.264 + AAC. At 2 points per clip, the clip frame decoded by ffmpeg matches the raw frame at the same time (diff ~2.4, H.264 loss) better than ±1 frame (6 to 18). The burned labels read correctly (e.g. `05:02.00 f9060`).
- `uv run pytest`: 55 passed. `git status`: nothing under `data/` or `outputs/`.

**Golden set:** n/a (picked by Ari after this section).

**Known issues:**
- Dissolves between rounds are missed, and slow-motion replays give fake cuts (see Open issues). Neither affects live fighting.
- Munhoz-O'Malley's one-frame cross-blends at cuts can land on either side of the cut. 1.2 should treat a cut as ±1 frame, e.g. by not starting new tracks on the blend frame.
- Short shots at the edges of the window (e.g. Garbrandt 00:40.00-00:40.37) are window slivers, not fake cuts.

**Next:**
- Ari: look at `outputs/shots_check/` (2 sheets + `report.md`) and pick the golden set from `outputs/golden_candidates/` (see ROADMAP 1.1 "You do").
- Then section 1.2 (detection and tracking).

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
