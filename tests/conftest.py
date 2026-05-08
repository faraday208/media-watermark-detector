"""Test fixture'ları — media-watermark-detector.

YOLO model olmadığında bile çalışsın diye mock-tabanlı.
Gerçek inference test'leri için ayrıca `pytest -m yolo` (model gerekli).
"""
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _save_jpg(p: Path, size=(256, 256), color="red") -> None:
    Image.new("RGB", size, color).save(p, "JPEG", quality=85)


@pytest.fixture
def mixed_dataset(tmp_path: Path) -> Path:
    """5 dosya: 3 top-level + 2 alt klasör."""
    _save_jpg(tmp_path / "a.jpg", color="red")
    _save_jpg(tmp_path / "b.jpg", color="blue")
    _save_jpg(tmp_path / "c.jpg", color="green")
    sub = tmp_path / "sub"
    sub.mkdir()
    _save_jpg(sub / "d.jpg", color="yellow")
    _save_jpg(sub / "e.jpg", color="magenta")
    return tmp_path


@pytest.fixture
def mock_scan_result(tmp_path: Path):
    """Sentetik ScanResult — YOLO çağırmadan apply_action / undo test eder."""
    from watermark_core.scanner import ScanResult

    # 3 dosya yarat (apply_action path resolve için gerek)
    files = []
    for name in ("with_wm_1.jpg", "with_wm_2.jpg", "clean.jpg"):
        p = tmp_path / name
        _save_jpg(p)
        files.append(p)

    # 2 watermark'lı + 1 temiz
    results = [
        {
            "valid": False, "reason": "watermark_detected (1)",
            "filename": "with_wm_1.jpg", "path": str(files[0].resolve()),
            "has_watermark": True, "detection_count": 1,
            "detections": [{"confidence": 0.85,
                            "bbox": [10, 20, 100, 80], "class_id": 0}],
            "error": None,
        },
        {
            "valid": False, "reason": "watermark_detected (2)",
            "filename": "with_wm_2.jpg", "path": str(files[1].resolve()),
            "has_watermark": True, "detection_count": 2,
            "detections": [
                {"confidence": 0.91, "bbox": [5, 5, 50, 30], "class_id": 0},
                {"confidence": 0.72, "bbox": [120, 100, 200, 180], "class_id": 0},
            ],
            "error": None,
        },
        {
            "valid": True, "reason": None,
            "filename": "clean.jpg", "path": str(files[2].resolve()),
            "has_watermark": False, "detection_count": 0,
            "detections": [], "error": None,
        },
    ]

    return ScanResult(
        source_root=str(tmp_path),
        total_scanned=3,
        valid_count=1,
        invalid_count=2,
        confidence_threshold=0.25,
        model_path="/mock/model.pt",
        results=results,
    )
