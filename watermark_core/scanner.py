"""
Watermark Scanner — collect_images + YOLO inference üzerinden watermark detection.

Public API:
- collect_images(directory, recursive, allowed_exts) → list[Path]
- find_watermarks(directory, *, model_path, confidence, recursive, ...) → ScanResult
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

DEFAULT_IMAGE_EXTS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"
})

DEFAULT_MODEL_PATH = os.path.expanduser(
    "~/Models/watermarks_yolov8/watermarks_s_yolov8_v1.pt"
)
DEFAULT_CONFIDENCE = 0.25

ProgressCallback = Callable[[int, int, str], None]


@dataclass
class WatermarkDetection:
    """Tek bir bbox detection — UI'da preview için bbox bilgisi."""
    confidence: float
    bbox: list[float]    # [x1, y1, x2, y2]
    class_id: int = 0    # YOLO class

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WatermarkResult:
    """Bir dosyanın watermark detection sonucu.

    valid=True → watermark YOK (temiz dosya)
    valid=False → watermark VAR, reason="watermark_detected (N detection)"
    """
    valid: bool
    reason: str | None
    filename: str
    path: str            # absolute — apply_action için
    has_watermark: bool
    detection_count: int = 0
    detections: list[dict] = field(default_factory=list)
    error: str | None = None  # YOLO inference hatası

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScanResult:
    source_root: str
    total_scanned: int
    valid_count: int          # watermark'sız (temiz)
    invalid_count: int        # watermark var
    error_count: int = 0
    confidence_threshold: float = DEFAULT_CONFIDENCE
    model_path: str = ""
    results: list[dict] = field(default_factory=list)

    @property
    def has_invalid(self) -> bool:
        return self.invalid_count > 0


def collect_images(
    directory: Path | str,
    *,
    recursive: bool = True,
    allowed_exts: Iterable[str] = DEFAULT_IMAGE_EXTS,
) -> list[Path]:
    """Görsel dosyaları topla (validator/dedup/quality ile aynı pattern)."""
    root = Path(directory)
    if not root.is_dir():
        return []
    exts = {e.lower() for e in allowed_exts}
    out: list[Path] = []
    if recursive:
        for dirpath, _dn, filenames in os.walk(root, followlinks=False):
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix.lower() in exts:
                    out.append(p)
    else:
        for entry in root.iterdir():
            if entry.is_file() and entry.suffix.lower() in exts:
                out.append(entry)
    out.sort()
    return out


def find_watermarks(
    directory: Path | str,
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    confidence: float = DEFAULT_CONFIDENCE,
    recursive: bool = True,
    allowed_exts: Iterable[str] = DEFAULT_IMAGE_EXTS,
    progress_cb: ProgressCallback | None = None,
) -> ScanResult:
    """YOLOv8 model ile watermark detection. Watermark'lı dosyalar 'invalid'."""
    root = Path(directory).resolve()
    images = collect_images(root, recursive=recursive, allowed_exts=allowed_exts)
    total = len(images)

    if total == 0:
        return ScanResult(
            source_root=str(root), total_scanned=0,
            valid_count=0, invalid_count=0,
            confidence_threshold=confidence, model_path=model_path,
        )

    # YOLO lazy import — ultralytics büyük dependency, sadece scan'de yükle
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError(
            "ultralytics gerekli — `uv sync` ile yükleyin"
        ) from e

    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"YOLO model bulunamadı: {model_path}\n"
            "Modeli ~/Models/watermarks_yolov8/ altına koyun veya "
            "--model PATH ile farklı bir model belirtin."
        )

    if progress_cb:
        progress_cb(0, total, f"YOLO yükleniyor: {Path(model_path).name}")
    model = YOLO(model_path)

    results: list[dict] = []
    valid_count = invalid_count = error_count = 0

    for idx, img in enumerate(images, 1):
        try:
            yolo_results = model(str(img), conf=confidence, verbose=False)
            detections: list[dict] = []
            if yolo_results and yolo_results[0].boxes is not None:
                boxes = yolo_results[0].boxes
                for box in boxes:
                    detections.append(WatermarkDetection(
                        confidence=float(box.conf[0]),
                        bbox=[float(v) for v in box.xyxy[0].tolist()],
                        class_id=int(box.cls[0]),
                    ).to_dict())

            has_wm = len(detections) > 0
            wm_result = WatermarkResult(
                valid=not has_wm,
                reason=(f"watermark_detected ({len(detections)})" if has_wm else None),
                filename=img.name,
                path=str(img.resolve()),
                has_watermark=has_wm,
                detection_count=len(detections),
                detections=detections,
            )
            results.append(wm_result.to_dict())
            if has_wm:
                invalid_count += 1
            else:
                valid_count += 1
        except Exception as e:
            results.append(WatermarkResult(
                valid=False, reason=f"error: {e}",
                filename=img.name, path=str(img.resolve()),
                has_watermark=False, error=str(e),
            ).to_dict())
            error_count += 1

        if progress_cb and (idx % 10 == 0 or idx == total):
            progress_cb(idx, total, f"Detection: {idx}/{total}")

    return ScanResult(
        source_root=str(root),
        total_scanned=total,
        valid_count=valid_count,
        invalid_count=invalid_count,
        error_count=error_count,
        confidence_threshold=confidence,
        model_path=model_path,
        results=results,
    )
