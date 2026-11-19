import csv

from deepfoul.config import load_config, pick_profile, repo_root


def test_paths_resolve_inside_repo():
    cfg = load_config()
    root = repo_root()
    for p in cfg["paths"].values():
        assert p.is_relative_to(root)


def test_profile_by_vram():
    cfg = load_config()
    cfg["profile"] = "auto"
    assert pick_profile(cfg, 24.0) == "large"
    assert pick_profile(cfg, 8.0) == "small"
    assert pick_profile(cfg, 4.0) == "cpu"
    assert pick_profile(cfg, None) == "cpu"


def test_gpu_profiles_use_performance_pose():
    cfg = load_config()
    assert cfg["profiles"]["large"]["pose_mode"] == "performance"
    assert cfg["profiles"]["small"]["pose_mode"] == "performance"
    assert cfg["profiles"]["cpu"]["pose_mode"] == "lightweight"


def test_explicit_profile_wins():
    cfg = load_config()
    cfg["profile"] = "cpu"
    assert pick_profile(cfg, 24.0) == "cpu"


EVENT_COLS = ["event_id", "fight_id", "start_s", "end_s", "foul_type", "offender",
              "contact", "ref_stopped", "speed", "source", "notes"]
EXPECTED = {
    "fights.csv": ["fight_id", "event", "date", "red_fighter", "blue_fighter", "red_shorts",
                   "blue_shorts", "source_url", "split", "action_start_s", "action_end_s", "notes"],
    "events.csv": EVENT_COLS,
    "golden.csv": EVENT_COLS,
}


def test_label_csv_headers():
    labels = repo_root() / "labels"
    for name, cols in EXPECTED.items():
        with (labels / name).open(newline="", encoding="utf-8-sig") as f:
            assert next(csv.reader(f)) == cols, name
