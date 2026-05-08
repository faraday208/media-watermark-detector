#!/usr/bin/env python3
"""
Validation Sonuçlarını Training Data'ya Dönüştürücü

Validation edilen resimleri ve doğru tespit edenlerin label'larını
YOLO formatında dataset klasörüne hazırlar.
"""

import json
import shutil
import argparse
from pathlib import Path
from PIL import Image


def find_detection_report(reports_dir, model_name, scan_date):
    """
    Model adı ve tarihine göre detection report'u bul

    Args:
        reports_dir: Reports klasörü
        model_name: Model adı (örn: "watermarks_s_yolov8_v2_finetuned")
        scan_date: Scan tarihi (örn: "2025-11-17")

    Returns:
        Report JSON path veya None
    """
    reports_path = Path(reports_dir)

    # Tarih formatı: watermark_report_YYYYMMDD_HHMMSS.json
    date_prefix = scan_date.replace("-", "")  # 20251117

    # Tüm report'ları kontrol et
    for report_file in reports_path.glob(f"watermark_report_{date_prefix}_*.json"):
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                report_data = json.load(f)

            # Model path'inden model adını çıkar
            model_path = report_data.get('metadata', {}).get('model_used', '')
            if model_name in model_path:
                return report_file
        except Exception as e:
            print(f"  ⚠️  Error reading {report_file.name}: {e}")
            continue

    return None


def bbox_to_yolo(bbox, img_width, img_height):
    """
    Detection bbox'unu YOLO formatına dönüştür

    Args:
        bbox: [x1, y1, x2, y2] formatında bbox
        img_width: Resim genişliği
        img_height: Resim yüksekliği

    Returns:
        (x_center, y_center, width, height) normalized
    """
    x1, y1, x2, y2 = bbox

    # Merkez koordinatları
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    # Genişlik ve yükseklik
    width = x2 - x1
    height = y2 - y1

    # Normalize et (0-1 arası)
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    return x_center_norm, y_center_norm, width_norm, height_norm


def process_validation_results(validation_json, reports_dir, output_dir, skip_existing=False):
    """
    Validation sonuçlarını işle ve training data hazırla

    Args:
        validation_json: Validation JSON dosyası
        reports_dir: Detection report'ların bulunduğu klasör
        output_dir: Hedef dataset klasörü
        skip_existing: Varolan dosyaları atla
    """
    # Klasörleri oluştur
    images_dir = Path(output_dir) / "images"
    labels_dir = Path(output_dir) / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Validation JSON oku
    print(f"\n📂 Loading validation results: {validation_json}")
    with open(validation_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = data['metadata']
    validations = data['validations']

    print(f"   Total validations: {metadata['total_validations']}")
    print(f"   Reports compared: {len(metadata['reports_compared'])}")

    # İstatistikler
    stats = {
        'copied': 0,
        'labels_created': 0,
        'manual_needed': [],
        'skipped': 0,
        'errors': 0
    }

    # Her validation'ı işle
    print(f"\n🔄 Processing validations...\n")

    for idx, validation in enumerate(validations, 1):
        filename = validation['filename']
        filepath = validation['filepath']
        correct_idx = validation['correct_report_index']

        print(f"[{idx}/{len(validations)}] {filename}")

        # Resim dosyasını kontrol et
        if not Path(filepath).exists():
            print(f"  ❌ Source file not found: {filepath}")
            stats['errors'] += 1
            continue

        # Hedef resim yolu
        dest_image = images_dir / filename

        # Varolan dosya kontrolü
        if skip_existing and dest_image.exists():
            print(f"  ⏭️  Already exists, skipping")
            stats['skipped'] += 1
            continue

        # Resmi kopyala
        try:
            shutil.copy2(filepath, dest_image)
            stats['copied'] += 1
            print(f"  ✅ Copied to images/")
        except Exception as e:
            print(f"  ❌ Copy error: {e}")
            stats['errors'] += 1
            continue

        # Label oluştur (eğer doğru tespit varsa)
        if correct_idx >= 0:
            # Doğru model adını ve tarihini al
            correct_report_name = validation['report_names'][correct_idx]

            # Model adını parse et (örn: "Report 1: watermarks_s_yolov8_v2_finetuned (Conf: 0.25) - 2025-11-17")
            # Format: "Report X: MODEL_NAME (Conf: X.XX) - YYYY-MM-DD"
            try:
                parts = correct_report_name.split(" (Conf:")
                model_part = parts[0].split(": ", 1)[1]  # "watermarks_s_yolov8_v2_finetuned"
                date_part = parts[1].split(" - ")[1]  # "2025-11-17"

                # Detection report'u bul
                report_file = find_detection_report(reports_dir, model_part, date_part)

                if not report_file:
                    print(f"  ⚠️  Detection report not found for {model_part} on {date_part}")
                    print(f"      Label creation skipped, manual annotation needed")
                    stats['manual_needed'].append(filename)
                    continue

                # Report'u oku
                with open(report_file, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)

                # Bu resmin detection'ını bul
                detections = None
                for item in report_data['results']['with_watermark']:
                    if item['filename'] == filename:
                        detections = item['detections']
                        break

                if not detections:
                    print(f"  ⚠️  No detections found in report")
                    print(f"      Label creation skipped, manual annotation needed")
                    stats['manual_needed'].append(filename)
                    continue

                # Resim boyutlarını al
                img = Image.open(filepath)
                img_width, img_height = img.size
                img.close()

                # YOLO label dosyası oluştur
                label_file = labels_dir / f"{Path(filename).stem}.txt"

                with open(label_file, 'w') as f:
                    for detection in detections:
                        bbox = detection['bbox']
                        x_center, y_center, width, height = bbox_to_yolo(bbox, img_width, img_height)

                        # YOLO format: class x_center y_center width height
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                stats['labels_created'] += 1
                print(f"  📝 Label created ({len(detections)} bbox)")

            except Exception as e:
                print(f"  ❌ Label creation error: {e}")
                stats['manual_needed'].append(filename)
                continue

        elif correct_idx == -2:
            # Unsure - Her iki raporun bbox'larını birleştir
            print(f"  🔄 Unsure - Merging both reports' detections")

            try:
                # Her iki raporun bbox'larını topla
                all_detections = []

                for report_idx in range(len(validation['report_names'])):
                    report_name = validation['report_names'][report_idx]

                    # Model adını ve tarihini parse et
                    parts = report_name.split(" (Conf:")
                    model_part = parts[0].split(": ", 1)[1]
                    date_part = parts[1].split(" - ")[1]

                    # Detection report'u bul
                    report_file = find_detection_report(reports_dir, model_part, date_part)

                    if not report_file:
                        continue

                    # Report'u oku
                    with open(report_file, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)

                    # Bu resmin detection'ını bul
                    for item in report_data['results']['with_watermark']:
                        if item['filename'] == filename:
                            all_detections.extend(item['detections'])
                            break

                if not all_detections:
                    print(f"  ⚠️  No detections found in any report")
                    print(f"      Label creation skipped, manual annotation needed")
                    stats['manual_needed'].append(filename)
                    continue

                # Resim boyutlarını al
                img = Image.open(filepath)
                img_width, img_height = img.size
                img.close()

                # YOLO label dosyası oluştur (tüm bbox'lar)
                label_file = labels_dir / f"{Path(filename).stem}.txt"

                with open(label_file, 'w') as f:
                    for detection in all_detections:
                        bbox = detection['bbox']
                        x_center, y_center, width, height = bbox_to_yolo(bbox, img_width, img_height)

                        # YOLO format: class x_center y_center width height
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                stats['labels_created'] += 1
                print(f"  📝 Label created (merged {len(all_detections)} bbox from both reports)")

            except Exception as e:
                print(f"  ❌ Label merging error: {e}")
                stats['manual_needed'].append(filename)
                continue

        else:
            # Neither is correct (-1)
            print(f"  ℹ️  Neither - Manual annotation needed")
            stats['manual_needed'].append(filename)

    # Özet rapor
    print(f"\n{'='*60}")
    print(f"✅ Processing Complete!")
    print(f"{'='*60}\n")

    print(f"📊 Summary:")
    print(f"   Images copied: {stats['copied']}")
    print(f"   Labels created: {stats['labels_created']}")
    print(f"   Manual annotation needed: {len(stats['manual_needed'])}")
    print(f"   Skipped (already exists): {stats['skipped']}")
    print(f"   Errors: {stats['errors']}")

    print(f"\n📁 Output:")
    print(f"   Images: {images_dir.absolute()} ({len(list(images_dir.glob('*')))} total)")
    print(f"   Labels: {labels_dir.absolute()} ({len(list(labels_dir.glob('*.txt')))} total)")

    if stats['manual_needed']:
        print(f"\n⚠️  Manual Annotation Needed ({len(stats['manual_needed'])} images):")
        for filename in stats['manual_needed'][:20]:  # İlk 20 tanesini göster
            print(f"   - {filename}")
        if len(stats['manual_needed']) > 20:
            print(f"   ... and {len(stats['manual_needed']) - 20} more")

    print(f"\n🎯 Next Steps:")
    print(f"   1. Annotate {len(stats['manual_needed'])} images using Gradio Annotation tab")
    print(f"   2. Run split_dataset.py to create train/val split")
    print(f"   3. Run train_model.py to start training v2.1_incremental")


def main():
    parser = argparse.ArgumentParser(
        description="Convert validation results to YOLO training data"
    )

    parser.add_argument(
        '--validation-json',
        type=str,
        required=True,
        help='Path to validation results JSON file'
    )

    parser.add_argument(
        '--reports-dir',
        type=str,
        required=True,
        help='Directory containing detection reports'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output dataset directory (will create images/ and labels/ subdirs)'
    )

    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip files that already exist in output directory'
    )

    args = parser.parse_args()

    # Dosya kontrolü
    if not Path(args.validation_json).exists():
        print(f"❌ Error: Validation JSON not found: {args.validation_json}")
        return 1

    if not Path(args.reports_dir).exists():
        print(f"❌ Error: Reports directory not found: {args.reports_dir}")
        return 1

    # İşlemi başlat
    process_validation_results(
        args.validation_json,
        args.reports_dir,
        args.output_dir,
        args.skip_existing
    )

    return 0


if __name__ == "__main__":
    exit(main())
