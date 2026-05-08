"""run.py CLI: argparse + main()."""
import sys
from pathlib import Path

import pytest

from run import _build_parser, _resolve_report_path, main


def test_parser_defaults():
    args = _build_parser().parse_args(["-i", "/tmp/x"])
    assert args.input == "/tmp/x"
    assert args.recursive is True
    assert args.invalid_action == "none"
    assert 0 <= args.confidence <= 1


def test_parser_full_flags():
    args = _build_parser().parse_args([
        "-i", "/tmp/x",
        "--no-recursive",
        "--model", "/m.pt",
        "--confidence", "0.4",
        "--invalid-action", "move", "--invalid-dir", "/r",
        "--dry-run", "--yes",
    ])
    assert args.recursive is False
    assert args.model == "/m.pt"
    assert args.confidence == 0.4
    assert args.invalid_action == "move"


def test_resolve_report_path_explicit(tmp_path: Path):
    args = _build_parser().parse_args(["-i", str(tmp_path), "-o", "/tmp/r.json"])
    assert _resolve_report_path(args, tmp_path) == Path("/tmp/r.json")


def test_resolve_report_path_move_uses_invalid_dir(tmp_path: Path):
    args = _build_parser().parse_args([
        "-i", str(tmp_path),
        "--invalid-action", "move", "--invalid-dir", str(tmp_path / "rej"),
    ])
    out = _resolve_report_path(args, tmp_path)
    assert out == tmp_path / "rej" / "watermark_report.json"


def test_main_input_required(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run.py"])
    with pytest.raises(SystemExit):
        main()


def test_main_undo_mutex(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sys, "argv", [
        "run.py", "--undo", str(tmp_path / "x.json"), "-i", "/tmp/y",
    ])
    with pytest.raises(SystemExit):
        main()


def test_main_invalid_dir_required_for_move(monkeypatch, mixed_dataset: Path, tmp_path: Path):
    """Move action --invalid-dir gerektirir."""
    fake_model = tmp_path / "m.pt"; fake_model.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", [
        "run.py", "-i", str(mixed_dataset),
        "--model", str(fake_model),
        "--invalid-action", "move",
    ])
    with pytest.raises(SystemExit):
        main()


def test_main_missing_model_returns_1(monkeypatch, mixed_dataset: Path):
    monkeypatch.setattr(sys, "argv", [
        "run.py", "-i", str(mixed_dataset),
        "--model", "/nonexistent/model.pt",
    ])
    rc = main()
    assert rc == 1


def test_main_writes_report_with_mock_yolo(monkeypatch, mixed_dataset: Path, tmp_path: Path):
    """E2E mock'la: 5 dosya tarama + rapor yazma."""
    import json
    fake_model = tmp_path / "fake.pt"
    fake_model.write_bytes(b"")

    class FakeYOLO:
        def __init__(self, path): pass
        def __call__(self, p, conf=0.25, verbose=False):
            class R:
                boxes = None
            return [R()]

    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO)
    monkeypatch.setattr(sys, "argv", [
        "run.py", "-i", str(mixed_dataset),
        "--model", str(fake_model),
    ])
    rc = main()
    assert rc == 0
    report = mixed_dataset / "watermark_report.json"
    assert report.exists()
    data = json.loads(report.read_text())
    assert data["tool"] == "media-watermark-detector"
    assert data["summary"]["total_scanned"] == 5
