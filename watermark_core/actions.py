"""
Watermark action layer — watermark'lı dosyaları move/delete + sidecar JSON undo.
Validator pattern'i (path-based dosya çözümleme).
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

REPORT_VERSION = "1"
REPORT_TOOL = "media-watermark-detector"
DEFAULT_REPORT_NAME = "watermark_report.json"


@dataclass
class ActionEntry:
    original: str
    reason: str
    moved_to: str | None = None
    deleted: bool = False
    detection_count: int = 0

    def to_dict(self) -> dict:
        d = {
            "original": self.original,
            "reason": self.reason,
            "detection_count": self.detection_count,
        }
        if self.moved_to is not None:
            d["moved_to"] = self.moved_to
        if self.deleted:
            d["deleted"] = True
        return d


@dataclass
class ActionResult:
    action: str
    invalid_dir: str | None
    entries: list[ActionEntry] = field(default_factory=list)
    skipped: int = 0


def _unique_target(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def apply_action(
    results: list[dict],
    *,
    source_root: Path | str,
    action: str = "none",
    invalid_dir: Path | str | None = None,
    dry_run: bool = False,
) -> ActionResult:
    """Watermark'lı (invalid) dosyaları taşı/sil."""
    if action not in ("none", "move", "delete"):
        raise ValueError(f"action must be 'none'/'move'/'delete', got {action!r}")
    if action == "move" and invalid_dir is None:
        raise ValueError("action='move' requires invalid_dir")

    src_root = Path(source_root).resolve()
    dst_root = Path(invalid_dir).resolve() if invalid_dir else None
    result = ActionResult(
        action=action,
        invalid_dir=str(dst_root) if dst_root else None,
    )

    if action == "none":
        return result

    if action == "move" and dst_root and not dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)

    for r in results:
        if r.get("valid"):  # temiz dosya
            continue
        original = _resolve_path(r, src_root)
        if original is None:
            result.skipped += 1
            continue
        reason = r.get("reason") or "watermark"
        det_count = r.get("detection_count", 0)

        if action == "delete":
            if not dry_run:
                try:
                    original.unlink()
                except OSError:
                    result.skipped += 1
                    continue
            result.entries.append(ActionEntry(
                original=str(original), reason=reason,
                detection_count=det_count, deleted=True,
            ))
        elif action == "move" and dst_root:
            target = _unique_target(dst_root / original.name)
            if not dry_run:
                try:
                    shutil.move(str(original), str(target))
                except OSError:
                    result.skipped += 1
                    continue
            result.entries.append(ActionEntry(
                original=str(original), reason=reason,
                detection_count=det_count,
                moved_to=str(target),
            ))

    return result


def _resolve_path(r: dict, src_root: Path) -> Path | None:
    p = r.get("path") or ""
    if p:
        cand = Path(p)
        if cand.is_file():
            return cand
    name = r.get("filename") or ""
    if not name:
        return None
    direct = src_root / name
    if direct.is_file():
        return direct
    for hit in src_root.rglob(name):
        if hit.is_file():
            return hit
    return None


def undo_from_report(report_path: Path | str, *, dry_run: bool = False) -> dict:
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    if report.get("tool") != REPORT_TOOL:
        raise ValueError(
            f"Report tool mismatch: expected {REPORT_TOOL!r}, "
            f"got {report.get('tool')!r}"
        )
    restored = skipped = irreversible = 0
    for entry in report.get("actions", []):
        if entry.get("deleted"):
            irreversible += 1
            continue
        moved_to = entry.get("moved_to")
        original = entry.get("original")
        if not moved_to or not original:
            skipped += 1
            continue
        src = Path(moved_to)
        dst = Path(original)
        if not src.exists() or dst.exists():
            skipped += 1
            continue
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(src), str(dst))
            except OSError:
                skipped += 1
                continue
        restored += 1
    return {
        "restored": restored,
        "skipped": skipped,
        "irreversible_deletes": irreversible,
    }
