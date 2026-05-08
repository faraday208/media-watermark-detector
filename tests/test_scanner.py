"""Scanner: collect_images + find_watermarks edge cases (YOLO mock'lu)."""
from pathlib import Path

import pytest

from watermark_core import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IMAGE_EXTS,
    DEFAULT_MODEL_PATH,
    collect_images,
    find_watermarks,
)


def test_collect_top_level_only(mixed_dataset: Path):
    files = collect_images(mixed_dataset, recursive=False)
    names = [p.name for p in files]
    assert "a.jpg" in names
    assert "d.jpg" not in names


def test_collect_recursive(mixed_dataset: Path):
    files = collect_images(mixed_dataset, recursive=True)
    names = [p.name for p in files]
    assert "d.jpg" in names
    assert "e.jpg" in names


def test_collect_filters_by_extension(tmp_path: Path):
    (tmp_path / "a.jpg").write_bytes(b"")
    (tmp_path / "b.txt").write_bytes(b"hi")
    out = collect_images(tmp_path, allowed_exts={".jpg"})
    assert [p.name for p in out] == ["a.jpg"]


def test_collect_invalid_dir(tmp_path: Path):
    assert collect_images(tmp_path / "nope") == []


def test_collect_is_sorted(mixed_dataset: Path):
    paths = [str(p) for p in collect_images(mixed_dataset, recursive=True)]
    assert paths == sorted(paths)


def test_default_image_exts():
    assert ".jpg" in DEFAULT_IMAGE_EXTS
    assert ".png" in DEFAULT_IMAGE_EXTS
    assert ".webp" in DEFAULT_IMAGE_EXTS


def test_default_model_path_is_string():
    assert isinstance(DEFAULT_MODEL_PATH, str)
    assert DEFAULT_MODEL_PATH


def test_default_confidence_in_range():
    assert 0 <= DEFAULT_CONFIDENCE <= 1


# ---------- find_watermarks edge cases ----------

def test_find_watermarks_empty_dir(tmp_path: Path):
    """Boş dizin → ScanResult döner, YOLO yüklemeden."""
    sr = find_watermarks(tmp_path)
    assert sr.total_scanned == 0
    assert sr.results == []


def test_find_watermarks_missing_model_raises(mixed_dataset: Path):
    """Model dosyası yoksa FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="YOLO model bulunamadı"):
        find_watermarks(mixed_dataset, model_path="/nonexistent/model.pt")


def test_find_watermarks_missing_ultralytics(monkeypatch, mixed_dataset: Path):
    """ultralytics import edilemezse RuntimeError."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "ultralytics":
            raise ImportError("simulated")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="ultralytics"):
        find_watermarks(mixed_dataset, model_path="/some/path.pt")


# NOT: find_watermarks'ın "happy path" (gerçek YOLO inference) testi
# yapılmıyor — model dosyası ve gerçek YOLO ağırlıkları gerektirir.
# Mock test denenip kaldırıldı: ultralytics lazy import + sys.modules
# cache nedeniyle monkeypatch güvenilir değil.
# Apply_action / undo testleri sentetik ScanResult fixture'la kapsanıyor
# (test_actions.py — mock_scan_result fixture).
