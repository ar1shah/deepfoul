# DeepFoul Progress

Claude Code updates this file at the end of every ROADMAP section. Newest entries at the top of the log.

## Current status
| Phase | Section | Status |
|---|---|---|
| 0 | 0.1 Repo and environment | Done (RTX 2080) |
| 0 | 0.2 Starter fights | Done (15/15 downloaded) |
| 1 | 1.1 Video IO and shot detection | Done (shots for 9 train fights) |
| 1 | 1.2 Detection and tracking | Done (stand-up stable; clinch/ground to 1.3) |
| 1 | 1.3 Identity: ref filter and red/blue | In progress (paused after step 1, shorts colors) |
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
- Pipeline speed (fps on RTX 2080): person detection 20-23 fps (YOLOX-m, ~45 ms/frame, about half of it the in-model NMS). Tracking + 1080p decode + debug render: 13-16 fps end to end.
- Shot detection (CPU decode): ~190 fps at 1080p30, ~110 fps at 60 fps. All 9 train fights: ~38 min.
- Labeled positives: eye_poke - / low_blow - / open_hand -
- Fights: train 9 / val 4 / test 2 (8 foul, 5 clean, 2 test). Known fouls: 16 (`labels/notes/known_fouls.csv`)
- Val fights: Cortes Acosta-Delija, Kape-Silva, Gaethje-Holloway, Garry-Prates (6 known fouls). Kape-Silva and Garbrandt-Long swapped splits after 0.2, because Garbrandt-Long's video is cropped and likely frame-interpolated.
- Best val metrics: -
- Test metrics: -

## Open issues
- **Detection speed:** ~20 fps means ~35 min of detection per 25-min fight before pose. Fine overnight; options for 1.4/1.5 if needed: detect every 2nd frame, fp16, or a lower NMS candidate cap.
- **Clinch and ground tracking** (1.2): a tied-up pair is often one merged box, so IDs split in clinches (golden_04: 10-12 tracks per shot). On the ground, lying fighters have short boxes, so seated cageside officials pass the relative-height filter and get tracked. The ref gets a new ID each time he leaves and re-enters the frame. All for 1.3 (profiles + ref filter).
- openmmlab's model server throttles to ~100 KB/s per connection. On a new machine, pre-download the two model zips from rtmlib's Hugging Face mirror (see the 0.1 log entry).
- **6 unofficial uploads** (Aspinall-Gane, Battle-Loosa, Garbrandt-Long, Holland-Malott, Munhoz-O'Malley, Edwards-Muhammad). Accepted for now; ask Cowork for official replacements in a data round. They can vanish or be re-uploaded with a different cut, which would break every timestamp for that fight. If a URL is swapped, the fight's window and known fouls have to be re-timed.
- Edwards-Muhammad is 480p (under the 720p bar). Kept for train; drop it if pose is poor in 1.4.
- Garbrandt-Long is a Russian broadcast at 1694x1080, 60 fps. 1.1 found it's a **25 fps source padded with duplicate frames** (58% of frames repeat the previous one), not interpolated. Motion features (3.2) must not read a zero velocity on a duplicate frame as "stopped". Munhoz-O'Malley is 50 fps with one-frame cross-blends at cuts.
- Shot detection misses **dissolves** (gradual transitions). They only showed up in between-round segments (Garbrandt 12:16-13:36, Holland 20:13-21:13), which 4.2 skips.
- Slow-motion replays can fire bursts of fake cuts (Battle-Loosa 09:40-09:43, 4 shots under 1 s). Replays are tagged in 4.2.
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

### 2026-11-20 · Section 1.3 · Identity (paused at step 1, the shorts-colors pause point)
**Approved plan:** `C:\Users\AriSh\.claude\plans\read-claude-md-roadmap-md-and-dazzling-moth.md` (local to the 2080 PC). Its key points, so a fresh session can continue:
- **Identity from appearance, not carried IDs:** each kept box is matched against fight-level profiles (red / blue / ref), then labelled red, blue, ref, merged or none. ByteTrack tracklets are only used to smooth labels (±0.5 s majority vote, and a tracklet is split on a stable label flip).
- **Pose-guided regions** (shorts = hips to 40% of the way to the knees, torso, shins = knees to ankles, skin reference from the cheeks). RTMW runs **once per kept box** (never on filtered boxes), is cached, and is reused by 1.4. `--from-cache` re-runs identity without the GPU.
- **Ref:** mainly clothed shins (vs the person's own face skin) plus a learned ref profile. "Torso isn't skin" only counts for men (women fighters wear tops). Fallback: with exactly 3 people, the third is the ref.
- **Merged clinch box:** labelled `merged` (red+blue), not split.
- **Profiles:** seeded from the shorts color names, then refined on the first ~60 s of the fight. report.md warns if they drift far from the seed or look swapped.
- **`id-golden` command:** renders to `outputs/renders/1.3/`, nothing drawn on the ref unless `--show-ref` is passed. The report carries a per-clip ref check (golden 4/6 called out) and a table of Ari's 1.2 issues #2 to #8.
- **For 1.4:** record det + pose fps and how long one 15-min fight takes.

**Ari's 1.2 tracking issues** (original-video time):
- **#2** golden_01 08:12: the ref walks in front of Battle, takes his ID, and Battle gets a new one.
- **#3** golden_02 04:58: Holland and the ref swap IDs at a cut.
- **#4** golden_03 08:53: Vera and O'Malley swap mid-shot, then swap back at 09:09.
- **#5** golden_04 14:03: after the angle change the clinch is one merged box.
- **#6** golden_04 from 14:03: crowd through the fence gets IDs, Barber and Grasso merge and swap, and the ref swaps with Grasso while walking in front of the camera.
- **#7** golden_05 11:12: the ref gets a new ID after walking behind Shara.
- **#8** golden_06 08:41: the fighters and the ref cycle IDs after an angle change, and they're fixed again at the 08:48 angle change.

**Built so far (step 1):**
- `deepfoul shorts-check [ids] [--before 180 --after 60 --step 5]` writes contact sheets to `outputs/shorts_check/sheets/`. `--save fight_id=mm:ss` saves the chosen frame to `outputs/shorts_check/<fight_id>.jpg`. Both write `outputs/shorts_check/report.md`.
- `identity.colors` (named shorts colors as HSV ranges) and `identity.similar_pairs` in the config. `validate_fights(rows, colors)` checks the shorts names. Format: main color first, accents after `+`.
- `red_shorts` / `blue_shorts` filled for all 15 fights, mostly from the broadcast trunk-color graphic. Barber-Grasso (2021) has no graphic, so I read it from the fighters (glove trim gives the corner). In two fights the shorts color is the opposite of the corner color: Garbrandt-Long (red corner wears blue) and Holland-Malott (red corner wears black, blue corner wears red).
- **No truly similar-shorts fight.** The only flag is Barber-Grasso (light gray vs black), which is the stand-in for Done-when #2. It's also the women's fight, and the tops are reversed (Barber: black top + gray shorts, Grasso: white top + black shorts), so it tests that shorts are read from the right region. "Find a similar-shorts fight" is now a data-round task in ROADMAP.

**Next:** Ari checks `outputs/shorts_check/` and replies `colors ok` or sends corrections. Then steps 2 to 6 of the plan (pose, appearance, profiles and assignment, renders and report, tests).

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
