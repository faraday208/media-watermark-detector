"""media-watermark-detector — public API.

In-process kullanım için:
    from watermark_core import (
        find_watermarks, collect_images,
        apply_action, undo_from_report, write_report,
        WatermarkResult, ScanResult, WatermarkDetection,
    )
"""
from .actions import (
    ActionEntry,
    ActionResult,
    apply_action,
    undo_from_report,
)
from .reporter import (
    DEFAULT_REPORT_NAME,
    REPORT_TOOL,
    REPORT_VERSION,
    write_report,
)
from .scanner import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IMAGE_EXTS,
    DEFAULT_MODEL_PATH,
    ScanResult,
    WatermarkDetection,
    WatermarkResult,
    collect_images,
    find_watermarks,
)

__all__ = [
    # Scanner
    "find_watermarks",
    "collect_images",
    "WatermarkResult",
    "WatermarkDetection",
    "ScanResult",
    "DEFAULT_IMAGE_EXTS",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_CONFIDENCE",
    # Actions
    "apply_action",
    "undo_from_report",
    "ActionEntry",
    "ActionResult",
    # Reporter
    "write_report",
    "DEFAULT_REPORT_NAME",
    "REPORT_TOOL",
    "REPORT_VERSION",
]
