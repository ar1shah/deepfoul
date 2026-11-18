import subprocess

from deepfoul.cli import _tiny_add_model
from deepfoul.io.fetch import fetch_all, read_fights, ytdlp_command


def _fake_run(calls):
    def run(cmd):
        calls.append(cmd)
        out = cmd[cmd.index("-o") + 1]
        open(out, "wb").close()  # pretend yt-dlp wrote the file
        return subprocess.CompletedProcess(cmd, 0)
    return run


def test_skips_existing_and_downloads_missing(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "have.mp4").write_bytes(b"x")
    rows = [
        {"fight_id": "have", "source_url": "https://example.com/a"},
        {"fight_id": "need", "source_url": "https://example.com/b"},
        {"fight_id": "nourl", "source_url": ""},
    ]
    calls = []
    res = fetch_all(rows, raw, 1080, run=_fake_run(calls))
    assert res == {"downloaded": ["need"], "skipped": ["have"], "failed": ["nourl"]}
    assert len(calls) == 1 and calls[0][-1] == "https://example.com/b"


def test_only_filter(tmp_path):
    rows = [{"fight_id": "a", "source_url": "u1"}, {"fight_id": "b", "source_url": "u2"}]
    calls = []
    res = fetch_all(rows, tmp_path, 720, only="b", run=_fake_run(calls))
    assert res["downloaded"] == ["b"] and len(calls) == 1


def test_ytdlp_command_caps_height(tmp_path):
    cmd = ytdlp_command("URL", tmp_path / "x.mp4", 720)
    assert cmd[0] == "yt-dlp" and "height<=720" in cmd[2] and cmd[-1] == "URL"
    assert cmd[cmd.index("--merge-output-format") + 1] == "mp4"


def test_read_fights_handles_bom(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("fight_id,source_url\nabc,u\n,\n", encoding="utf-8-sig")
    assert read_fights(p) == [{"fight_id": "abc", "source_url": "u"}]


def test_tiny_onnx_model_runs_on_cpu():
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(_tiny_add_model(), providers=["CPUExecutionProvider"])
    (y,) = sess.run(None, {"x": np.array([1, 2, 3, 4], dtype=np.float32)})
    assert y.tolist() == [2, 4, 6, 8]
