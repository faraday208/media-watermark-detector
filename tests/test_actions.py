"""Actions: apply_action (move/delete) + undo. YOLO çağrılmadan mock_scan_result ile."""
from pathlib import Path

import pytest

from watermark_core import (
    apply_action,
    undo_from_report,
    write_report,
)


def test_action_none_no_changes(mock_scan_result, tmp_path: Path):
    sr = mock_scan_result
    ar = apply_action(sr.results, source_root=Path(sr.source_root), action="none")
    assert ar.entries == []
    # Tüm dosyalar yerinde
    assert Path(sr.source_root, "with_wm_1.jpg").exists()


def test_action_move_relocates_watermarked(mock_scan_result, tmp_path_factory):
    rejected = tmp_path_factory.mktemp("rejected")
    sr = mock_scan_result
    ar = apply_action(
        sr.results, source_root=Path(sr.source_root),
        action="move", invalid_dir=rejected,
    )
    assert ar.action == "move"
    # 2 watermark'lı taşındı
    assert len(ar.entries) == 2
    # Temiz yerinde
    assert Path(sr.source_root, "clean.jpg").exists()
    # Watermark'lılar /rejected'da
    moved_names = {Path(e.moved_to).name for e in ar.entries}
    assert "with_wm_1.jpg" in moved_names
    assert "with_wm_2.jpg" in moved_names


def test_action_move_dry_run_no_filesystem_change(mock_scan_result, tmp_path_factory):
    rejected = tmp_path_factory.mktemp("rejected")
    sr = mock_scan_result
    ar = apply_action(
        sr.results, source_root=Path(sr.source_root),
        action="move", invalid_dir=rejected, dry_run=True,
    )
    assert len(ar.entries) == 2
    # Yerinde kaldı
    assert Path(sr.source_root, "with_wm_1.jpg").exists()
    assert not list(rejected.iterdir())


def test_action_delete_removes_watermarked(mock_scan_result):
    sr = mock_scan_result
    ar = apply_action(sr.results, source_root=Path(sr.source_root), action="delete")
    assert all(e.deleted for e in ar.entries)
    assert not Path(sr.source_root, "with_wm_1.jpg").exists()
    assert Path(sr.source_root, "clean.jpg").exists()


def test_action_delete_dry_run_keeps_files(mock_scan_result):
    sr = mock_scan_result
    ar = apply_action(
        sr.results, source_root=Path(sr.source_root),
        action="delete", dry_run=True,
    )
    assert len(ar.entries) == 2
    assert Path(sr.source_root, "with_wm_1.jpg").exists()


def test_action_move_requires_invalid_dir(mock_scan_result):
    with pytest.raises(ValueError, match="invalid_dir"):
        apply_action(mock_scan_result.results,
                     source_root=Path(mock_scan_result.source_root),
                     action="move", invalid_dir=None)


def test_action_invalid_value_raises(mock_scan_result):
    with pytest.raises(ValueError, match="action"):
        apply_action(mock_scan_result.results,
                     source_root=Path(mock_scan_result.source_root),
                     action="burn")


def test_action_skips_unknown_filename(tmp_path: Path):
    res = apply_action(
        [{"valid": False, "filename": "ghost.jpg", "reason": "test", "path": ""}],
        source_root=tmp_path, action="delete",
    )
    assert res.entries == []
    assert res.skipped == 1


def test_action_entry_includes_detection_count(mock_scan_result, tmp_path_factory):
    rejected = tmp_path_factory.mktemp("rejected")
    sr = mock_scan_result
    ar = apply_action(
        sr.results, source_root=Path(sr.source_root),
        action="move", invalid_dir=rejected,
    )
    # detection_count rapora yansıyor
    counts = sorted(e.detection_count for e in ar.entries)
    assert counts == [1, 2]  # mock_scan_result'a göre


# ---------- Undo ----------

def test_undo_restores_moved(mock_scan_result, tmp_path_factory):
    rejected = tmp_path_factory.mktemp("rejected")
    sr = mock_scan_result
    ar = apply_action(
        sr.results, source_root=Path(sr.source_root),
        action="move", invalid_dir=rejected,
    )
    report = rejected / "watermark_report.json"
    write_report(report, scan_result=sr, action_result=ar, recursive=True)
    summary = undo_from_report(report)
    assert summary["restored"] == 2
    assert Path(sr.source_root, "with_wm_1.jpg").exists()


def test_undo_dry_run_no_changes(mock_scan_result, tmp_path_factory):
    rejected = tmp_path_factory.mktemp("rejected")
    sr = mock_scan_result
    ar = apply_action(
        sr.results, source_root=Path(sr.source_root),
        action="move", invalid_dir=rejected,
    )
    report = rejected / "watermark_report.json"
    write_report(report, scan_result=sr, action_result=ar, recursive=True)
    summary = undo_from_report(report, dry_run=True)
    assert summary["restored"] == 2
    # Hala rejected'ta
    assert not Path(sr.source_root, "with_wm_1.jpg").exists()


def test_undo_irreversible_for_delete(mock_scan_result, tmp_path: Path):
    sr = mock_scan_result
    ar = apply_action(sr.results, source_root=Path(sr.source_root), action="delete")
    report = tmp_path / "report.json"
    write_report(report, scan_result=sr, action_result=ar, recursive=True)
    summary = undo_from_report(report)
    assert summary["irreversible_deletes"] == 2
    assert summary["restored"] == 0


def test_undo_rejects_wrong_tool(tmp_path: Path):
    import json
    report = tmp_path / "fake.json"
    report.write_text(json.dumps({"tool": "other-tool", "actions": []}))
    with pytest.raises(ValueError, match="tool mismatch"):
        undo_from_report(report)
