#!/usr/bin/env python3
"""
Media Watermark Detector — CLI

Kullanım örnekleri:
  # Tarama (default model, sadece raporla)
  python run.py -i ./dataset

  # Confidence eşiği + model
  python run.py -i ./dataset --confidence 0.3 --model /path/to/model.pt

  # Watermark'lıları /rejected'a taşı
  python run.py -i ./dataset --invalid-action move --invalid-dir ./rejected

  # Geri al
  python run.py --undo ./rejected/watermark_report.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from watermark_core import (
    DEFAULT_CONFIDENCE,
    DEFAULT_MODEL_PATH,
    DEFAULT_REPORT_NAME,
    apply_action,
    find_watermarks,
    undo_from_report,
    write_report,
)


def _print_progress(current: int, total: int, msg: str) -> None:
    if total > 0:
        pct = current * 100 // total
        print(f"\r  {msg} ({pct}%)", end="", flush=True)
    else:
        print(f"\r  {msg}", end="", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Media Watermark Detector — YOLOv8 tabanlı watermark tespit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-i", "--input", help="Input klasörü (zorunlu, --undo hariç)")
    p.add_argument("-o", "--output", help=f"JSON rapor (default: <input>/{DEFAULT_REPORT_NAME})")
    p.add_argument("--recursive", action="store_true", default=True,
                   help="Alt klasörleri tara (default: True)")
    p.add_argument("--no-recursive", action="store_false", dest="recursive",
                   help="Sadece üst seviye")
    p.add_argument("--model", default=DEFAULT_MODEL_PATH,
                   help=f"YOLO model path (default: {DEFAULT_MODEL_PATH})")
    p.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
                   help=f"Detection confidence eşiği (default: {DEFAULT_CONFIDENCE})")
    p.add_argument("--limit", type=int, default=0, help="Max dosya")

    p.add_argument("--invalid-action", choices=["none", "move", "delete"],
                   default="none",
                   help="Watermark'lı dosyalar için aksiyon (default: none)")
    p.add_argument("--invalid-dir", help="--invalid-action move hedefi")
    p.add_argument("--dry-run", action="store_true",
                   help="Aksiyonu simüle et, dosyaya dokunma")
    p.add_argument("--yes", action="store_true",
                   help="Onay sorma (delete için)")

    p.add_argument("--undo", help="Watermark raporundan geri al (move action)")
    return p


def _confirm_delete(invalid: int, *, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    print(f"\n⚠  {invalid} watermark'lı dosya KALICI olarak silinecek. Geri alınamaz.")
    answer = input("Devam? [y/N]: ").strip().lower()
    return answer in {"y", "yes", "evet"}


def _run_undo(args: argparse.Namespace) -> int:
    report_path = Path(args.undo)
    if not report_path.exists():
        print(f"Rapor bulunamadı: {report_path}", file=sys.stderr)
        return 1
    print(f"Undo (dry-run={args.dry_run}): {report_path}")
    summary = undo_from_report(report_path, dry_run=args.dry_run)
    print(f"  Restored:               {summary['restored']}")
    print(f"  Skipped:                {summary['skipped']}")
    print(f"  Irreversible (deleted): {summary['irreversible_deletes']}")
    if summary["irreversible_deletes"]:
        print("  → Silinen geri getirilemez.")
    return 0


def _resolve_report_path(args: argparse.Namespace, input_dir: Path) -> Path:
    if args.output:
        return Path(args.output)
    if args.invalid_action == "move" and args.invalid_dir:
        return Path(args.invalid_dir) / DEFAULT_REPORT_NAME
    return input_dir / DEFAULT_REPORT_NAME


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.undo:
        if args.input or args.invalid_action != "none":
            parser.error("--undo ile -i/--input veya --invalid-action birlikte kullanılamaz")
        return _run_undo(args)

    if not args.input:
        parser.error("--input gerekli (veya --undo kullan)")

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Geçerli dizin değil: {input_dir}", file=sys.stderr)
        return 1

    if args.invalid_action == "move" and not args.invalid_dir:
        parser.error("--invalid-action move için --invalid-dir gerekli")

    print(f"\n{'='*70}")
    print(f"Media Watermark Detector")
    print(f"{'='*70}")
    print(f"Input:       {input_dir}")
    print(f"Recursive:   {args.recursive}")
    print(f"Model:       {args.model}")
    print(f"Confidence:  {args.confidence}")
    print(f"Action:      {args.invalid_action}{' (DRY-RUN)' if args.dry_run else ''}")
    print(f"{'='*70}\n")

    try:
        sr = find_watermarks(
            input_dir,
            model_path=args.model,
            confidence=args.confidence,
            recursive=args.recursive,
            progress_cb=_print_progress,
        )
    except FileNotFoundError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1

    if args.limit > 0:
        sr.results = sr.results[: args.limit]

    print(f"\n\n{'='*70}\nSONUÇLAR\n{'='*70}")
    print(f"Total scanned:    {sr.total_scanned}")
    print(f"Temiz (valid):    {sr.valid_count}")
    print(f"Watermark'lı:     {sr.invalid_count}")
    if sr.error_count:
        print(f"Errors:           {sr.error_count}")

    # Aksiyon
    if args.invalid_action == "delete" and sr.invalid_count > 0 and not args.dry_run:
        if not _confirm_delete(sr.invalid_count, assume_yes=args.yes):
            print("İptal edildi.")
            return 2

    ar = apply_action(
        sr.results,
        source_root=input_dir,
        action=args.invalid_action,
        invalid_dir=args.invalid_dir,
        dry_run=args.dry_run,
    )

    if args.invalid_action != "none":
        print(f"\nAksiyon: {args.invalid_action}{' (DRY-RUN)' if args.dry_run else ''}")
        if ar.action == "move":
            print(f"  Taşınan: {len(ar.entries)} → {ar.invalid_dir}")
        elif ar.action == "delete":
            print(f"  Silinen: {len(ar.entries)}")
        if ar.skipped:
            print(f"  Atlanan: {ar.skipped}")

    report_path = _resolve_report_path(args, input_dir)
    write_report(
        report_path,
        scan_result=sr,
        action_result=ar,
        recursive=args.recursive,
    )
    print(f"\nRapor: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
