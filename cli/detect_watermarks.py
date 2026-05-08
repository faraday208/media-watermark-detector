#!/usr/bin/env python3
"""
Watermark Detection Tool
Detects watermarks in images using YOLOv8 model
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: Required libraries not found!")
    print("Please install dependencies:")
    print("  pip install ultralytics pillow")
    sys.exit(1)


# Default model path
DEFAULT_MODEL_PATH = os.path.expanduser("~/Models/watermarks_yolov8/watermarks_s_yolov8_v1.pt")

# Supported image formats
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}


def scan_images(directory):
    """
    Scan directory for image files

    Returns:
        list of image file paths
    """
    images = []

    try:
        all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    except Exception as e:
        print(f"Error reading directory: {e}")
        return []

    for filename in all_files:
        ext = Path(filename).suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            filepath = os.path.join(directory, filename)
            images.append(filepath)

    return images


def detect_watermarks(model, image_paths, confidence_threshold=0.25):
    """
    Detect watermarks in images

    Args:
        model: YOLO model instance
        image_paths: list of image paths
        confidence_threshold: minimum confidence for detection

    Returns:
        dict: detection results
    """
    results_data = {
        'with_watermark': [],
        'without_watermark': [],
        'errors': []
    }

    print(f"\nProcessing {len(image_paths)} images...")
    print("=" * 80)

    for idx, img_path in enumerate(image_paths, 1):
        filename = os.path.basename(img_path)
        print(f"[{idx}/{len(image_paths)}] Processing: {filename}")

        try:
            # Run detection
            results = model(img_path, conf=confidence_threshold, verbose=False)

            # Check if watermark detected
            has_watermark = False
            detections = []

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    has_watermark = True

                    # Extract detection details
                    for box in boxes:
                        detection = {
                            'confidence': float(box.conf[0]),
                            'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                            'class': int(box.cls[0])
                        }
                        detections.append(detection)

            # Store result
            result_entry = {
                'filepath': img_path,
                'filename': filename,
                'has_watermark': has_watermark,
                'detections': detections,
                'detection_count': len(detections)
            }

            if has_watermark:
                results_data['with_watermark'].append(result_entry)
                print(f"  ✓ Watermark detected ({len(detections)} instance(s))")
            else:
                results_data['without_watermark'].append(result_entry)
                print(f"  ✗ No watermark")

        except Exception as e:
            print(f"  ERROR: {e}")
            results_data['errors'].append({
                'filepath': img_path,
                'filename': filename,
                'error': str(e)
            })

    return results_data


def save_reports(results, output_dir, directory_name, total_found=None, limit=None):
    """
    Save detection results to JSON and TXT reports
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure reports directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare report data
    processed_images = len(results['with_watermark']) + len(results['without_watermark'])
    watermark_count = len(results['with_watermark'])
    clean_count = len(results['without_watermark'])
    error_count = len(results['errors'])

    report_data = {
        'metadata': {
            'scan_date': datetime.now().isoformat(),
            'scanned_directory': directory_name,
            'total_images_found': total_found if total_found else processed_images,
            'images_processed': processed_images,
            'limit_applied': limit,
            'images_with_watermark': watermark_count,
            'images_without_watermark': clean_count,
            'errors': error_count
        },
        'results': results
    }

    # Save JSON report
    json_file = os.path.join(output_dir, f"watermark_report_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    # Save TXT report
    txt_file = os.path.join(output_dir, f"watermark_report_{timestamp}.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("WATERMARK DETECTION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Scan Date: {report_data['metadata']['scan_date']}\n")
        f.write(f"Directory: {directory_name}\n")
        f.write(f"Total Images Found: {report_data['metadata']['total_images_found']}\n")
        f.write(f"Images Processed: {processed_images}\n")
        if limit:
            f.write(f"Limit Applied: {limit}\n")
        f.write(f"Images with Watermark: {watermark_count}\n")
        f.write(f"Images without Watermark: {clean_count}\n")
        f.write(f"Errors: {error_count}\n\n")

        # Images with watermark
        if watermark_count > 0:
            f.write("=" * 80 + "\n")
            f.write("IMAGES WITH WATERMARK\n")
            f.write("=" * 80 + "\n\n")
            for item in results['with_watermark']:
                f.write(f"File: {item['filename']}\n")
                f.write(f"Path: {item['filepath']}\n")
                f.write(f"Detections: {item['detection_count']}\n")
                for det in item['detections']:
                    f.write(f"  - Confidence: {det['confidence']:.2%}\n")
                f.write("\n")

        # Images without watermark
        if clean_count > 0:
            f.write("=" * 80 + "\n")
            f.write("IMAGES WITHOUT WATERMARK\n")
            f.write("=" * 80 + "\n\n")
            for item in results['without_watermark']:
                f.write(f"{item['filename']}\n")

        # Errors
        if error_count > 0:
            f.write("\n" + "=" * 80 + "\n")
            f.write("ERRORS\n")
            f.write("=" * 80 + "\n\n")
            for item in results['errors']:
                f.write(f"File: {item['filename']}\n")
                f.write(f"Error: {item['error']}\n\n")

    print(f"\n\nReports saved:")
    print(f"  - {json_file}")
    print(f"  - {txt_file}")

    return json_file, txt_file


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 detect_watermarks.py <directory> [OPTIONS]")
        print("\nArguments:")
        print("  directory              : Directory containing images to scan")
        print("\nOptions:")
        print("  --model <path>         : Path to YOLOv8 model file")
        print(f"                           (default: {DEFAULT_MODEL_PATH})")
        print("  --confidence <float>   : Confidence threshold (0.0-1.0, default: 0.25)")
        print("  --limit <number>       : Process only first N images (default: all)")
        print("\nExamples:")
        print('  python3 detect_watermarks.py "/path/to/images"')
        print('  python3 detect_watermarks.py "/path/to/images" --confidence 0.5')
        print('  python3 detect_watermarks.py "/path/to/images" --limit 10')
        sys.exit(1)

    # Parse arguments
    target_dir = sys.argv[1]
    model_path = DEFAULT_MODEL_PATH
    confidence = 0.25
    limit = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--model' and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--confidence' and i + 1 < len(sys.argv):
            confidence = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--limit' and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    # Validate directory
    if not os.path.exists(target_dir):
        print(f"Error: Directory not found: {target_dir}")
        sys.exit(1)

    if not os.path.isdir(target_dir):
        print(f"Error: Not a directory: {target_dir}")
        sys.exit(1)

    # Validate model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    print()
    print("=" * 80)
    print("WATERMARK DETECTION")
    print("=" * 80)
    print(f"Directory: {target_dir}")
    print(f"Model: {model_path}")
    print(f"Confidence Threshold: {confidence}")
    if limit:
        print(f"Limit: {limit} images")
    else:
        print(f"Limit: All images")
    print()

    # Load model
    print("Loading YOLOv8 model...")
    try:
        model = YOLO(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Scan for images
    print("\nScanning for images...")
    all_image_paths = scan_images(target_dir)

    if not all_image_paths:
        print("No images found!")
        sys.exit(0)

    total_found = len(all_image_paths)
    print(f"Found {total_found} images")

    # Apply limit if specified
    if limit and limit < total_found:
        image_paths = all_image_paths[:limit]
        print(f"Processing first {limit} images (limit applied)")
    else:
        image_paths = all_image_paths

    # Detect watermarks
    results = detect_watermarks(model, image_paths, confidence)

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total images: {len(image_paths)}")
    print(f"With watermark: {len(results['with_watermark'])}")
    print(f"Without watermark: {len(results['without_watermark'])}")
    print(f"Errors: {len(results['errors'])}")

    # Save reports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(script_dir, "reports")
    save_reports(results, reports_dir, target_dir, total_found=total_found, limit=limit)

    print("\nDone!")


if __name__ == "__main__":
    main()
