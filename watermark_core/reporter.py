"""
Reporter — sidecar JSON (tool-conventions §4 uyumlu).

{
  "version": "1",
  "tool": "media-watermark-detector",
  "source_root": "/abs/path",
  "recursive": bool,
  "model_path": "...",
  "confidence_threshold": 0.25,
  "summary": {total_scanned, valid (clean), invalid (watermark), errors},
  "action": ...,
  "invalid_dir": ...,
  "actions": [...],
  "skipped": int,
  "results": [...]
}
"""
from __future__ import annotations

import json
from pathlib import Path

from .actions import ActionResult
from .scanner import ScanResult

REPORT_VERSION = "1"
REPORT_TOOL = "media-watermark-detector"
DEFAULT_REPORT_NAME = "watermark_report.json"


def write_report(
    report_path: Path | str,
    *,
    scan_result: ScanResult,
    action_result: ActionResult,
    recursive: bool,
) -> Path:
    summary = {
        "total_scanned": scan_result.total_scanned,
        "valid": scan_result.valid_count,         # watermark'sız
        "invalid": scan_result.invalid_count,     # watermark'lı
        "errors": scan_result.error_count,
    }
    payload = {
        "version": REPORT_VERSION,
        "tool": REPORT_TOOL,
        "source_root": scan_result.source_root,
        "recursive": recursive,
        "model_path": scan_result.model_path,
        "confidence_threshold": scan_result.confidence_threshold,
        "summary": summary,
        "action": action_result.action,
        "invalid_dir": action_result.invalid_dir,
        "actions": [e.to_dict() for e in action_result.entries],
        "skipped": action_result.skipped,
        "results": scan_result.results,
    }
    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out
