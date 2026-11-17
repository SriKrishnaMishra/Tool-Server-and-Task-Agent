import json

from bodhan_agent.__main__ import build_parser
from bodhan_agent.agent import preflight, run_task


def test_preflight_rejects_video_translation(tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not decoded")
    result = preflight("Translate this video into French", source.parent)
    assert not result.ok
    assert result.code == "unsupported_task"


def test_run_missing_files_writes_honest_layout(tmp_path):
    output = tmp_path / "out"
    assert run_task("translate the document", tmp_path / "missing", output) == 1
    assert (output / "result.md").exists()
    assert (output / "artifacts").is_dir()
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "impossible"
    assert list((output / "artifacts").iterdir()) == []
    assert json.loads((output / "trace.jsonl").read_text().splitlines()[0])["kind"] == "plan"


def test_parser_validates_negative_budget():
    parser = build_parser()
    try:
        parser.parse_args(["--task", "x", "--files", ".", "--out", "out", "--max-steps", "-1"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("negative budget should be rejected")
