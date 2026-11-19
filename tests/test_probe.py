from deepfoul.io.probe import bucket, flags_for, sheet_times, summarize, write_report


def _probe(duration="1000.0", height=1080, r="30/1", avg="30/1"):
    return {"format": {"duration": duration},
            "streams": [{"codec_type": "audio"},
                        {"codec_type": "video", "codec_name": "h264", "width": 1920,
                         "height": height, "r_frame_rate": r, "avg_frame_rate": avg}]}


ROW = {"fight_id": "ufc1_a_b", "action_start_s": "40", "action_end_s": "900",
       "notes": "bucket=foul; eye_poke"}


def test_summarize_picks_video_stream():
    info = summarize(_probe(r="30000/1001", avg="30000/1001"))
    assert info["height"] == 1080 and info["codec"] == "h264"
    assert abs(info["avg_fps"] - 29.97) < 0.01 and info["duration_s"] == 1000.0


def test_clean_video_has_no_flags():
    assert flags_for(ROW, summarize(_probe())) == []


def test_short_fight_with_window_is_fine():
    # Jones-Gane: a 3 min video is fine when the window fits inside it.
    row = {**ROW, "action_start_s": "43", "action_end_s": "196"}
    assert flags_for(row, summarize(_probe(duration="508"))) == []


def test_video_shorter_than_window():
    assert "shorter than window end" in flags_for(ROW, summarize(_probe(duration="800")))[0]


def test_length_check_only_without_window():
    row = {**ROW, "action_start_s": "", "action_end_s": ""}
    assert "odd length" in flags_for(row, summarize(_probe(duration="120")))[0]
    assert "odd length" in flags_for(row, summarize(_probe(duration="3600")))[0]
    assert flags_for(row, summarize(_probe(duration="1000"))) == []


def test_low_res_vfr_and_unofficial():
    row = {**ROW, "notes": "bucket=foul; unofficial upload (X)"}
    flags = flags_for(row, summarize(_probe(height=480, r="60/1", avg="29.5")))
    assert any("under 720p" in f for f in flags)
    assert any("variable frame rate" in f for f in flags)
    assert "unofficial upload" in flags


def test_sheet_times_use_window():
    assert sheet_times(ROW, 1000) == [40 + 0.1 * 860, 40 + 0.5 * 860, 40 + 0.9 * 860]
    assert sheet_times({"fight_id": "x"}, 1000) == [100, 500, 900]


def test_bucket_and_report(tmp_path):
    assert bucket(ROW) == "foul" and bucket({"notes": ""}) == "-"
    results = [
        {"fight_id": "a", "split": "train", "bucket": "foul", "status": "downloaded",
         "info": summarize(_probe()), "flags": [], "sheet": "a.jpg"},
        {"fight_id": "b", "split": "val", "bucket": "foul", "status": "failed",
         "info": {}, "flags": [], "sheet": ""},
    ]
    p = tmp_path / "report.md"
    write_report(results, p)
    text = p.read_text()
    assert "| a | train | foul | downloaded | 16:40 | 1920x1080 | 30.00 | ok | [a.jpg](a.jpg) |" in text
    assert "| foul | 1 | 1 | 0 | 0 |" in text
