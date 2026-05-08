#!/usr/bin/env python3
"""
Copy Clean Images
Copies images without watermarks to a separate folder based on detection report
"""

import os
import sys
import json
import shutil
from pathlib import Path


def load_report(report_path):
    """
    Load detection report from JSON file

    Returns:
        dict: Report data
    """
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading report: {e}")
        sys.exit(1)


def copy_clean_images(report_data, output_folder_name='no_watermarks'):
    """
    Copy images without watermarks to a separate folder

    Args:
        report_data: Detection report data
        output_folder_name: Name of the output folder (default: no_watermarks)

    Returns:
        tuple: (success_count, error_count, output_dir)
    """
    # Get scanned directory from metadata
    scanned_dir = report_data['metadata'].get('scanned_directory')
    if not scanned_dir or not os.path.exists(scanned_dir):
        print(f"Error: Scanned directory not found: {scanned_dir}")
        sys.exit(1)

    # Create output directory inside scanned directory
    output_dir = os.path.join(scanned_dir, output_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanned directory: {scanned_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Get list of images without watermark
    clean_images = report_data['results'].get('without_watermark', [])

    if not clean_images:
        print("No images without watermark found in report!")
        return 0, 0, output_dir

    print(f"Found {len(clean_images)} images without watermark")
    print("=" * 80)

    success_count = 0
    error_count = 0

    for idx, image_info in enumerate(clean_images, 1):
        source_path = image_info['filepath']
        filename = image_info['filename']

        print(f"[{idx}/{len(clean_images)}] Copying: {filename}")

        try:
            # Check if source file exists
            if not os.path.exists(source_path):
                print(f"  ✗ Source file not found: {source_path}")
                error_count += 1
                continue

            # Destination path
            dest_path = os.path.join(output_dir, filename)

            # Check if destination already exists
            if os.path.exists(dest_path):
                print(f"  ⚠ File already exists, skipping: {filename}")
                continue

            # Copy file
            shutil.copy2(source_path, dest_path)
            success_count += 1
            print(f"  ✓ Copied successfully")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            error_count += 1

    return success_count, error_count, output_dir


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 copy_clean_images.py <report.json> [output_folder_name]")
        print("\nArguments:")
        print("  report.json           : Path to watermark detection report (JSON)")
        print("  output_folder_name    : Name of output folder (default: no_watermarks)")
        print("\nExamples:")
        print('  python3 copy_clean_images.py reports/watermark_report_20251116_134131.json')
        print('  python3 copy_clean_images.py reports/watermark_report_20251116_134131.json clean_images')
        sys.exit(1)

    # Parse arguments
    report_path = sys.argv[1]
    output_folder_name = sys.argv[2] if len(sys.argv) > 2 else 'no_watermarks'

    # Validate report file
    if not os.path.exists(report_path):
        print(f"Error: Report file not found: {report_path}")
        sys.exit(1)

    print()
    print("=" * 80)
    print("COPY CLEAN IMAGES")
    print("=" * 80)
    print(f"Report: {report_path}")
    print(f"Output folder name: {output_folder_name}")
    print()

    # Load report
    print("Loading report...")
    report_data = load_report(report_path)

    # Get metadata
    total_images = report_data['metadata'].get('total_images_found', 0)
    with_watermark = report_data['metadata'].get('images_with_watermark', 0)
    without_watermark = report_data['metadata'].get('images_without_watermark', 0)

    print(f"Total images scanned: {total_images}")
    print(f"With watermark: {with_watermark}")
    print(f"Without watermark: {without_watermark}")
    print()

    # Copy images
    success_count, error_count, output_dir = copy_clean_images(
        report_data,
        output_folder_name
    )

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Successfully copied: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Output directory: {output_dir}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
