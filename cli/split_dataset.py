#!/usr/bin/env python3
"""
Split dataset into train/val sets
"""

import argparse
import shutil
import random
from pathlib import Path


def split_dataset(dataset_dir: Path, train_ratio: float = 0.8, seed: int = 42):
    """
    Split dataset into train/val sets

    Args:
        dataset_dir: Path to dataset directory containing images/ and labels/
        train_ratio: Ratio of training data (default: 0.8 for 80/20 split)
        seed: Random seed for reproducibility
    """
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    train_images_dir = dataset_dir / "train" / "images"
    train_labels_dir = dataset_dir / "train" / "labels"
    val_images_dir = dataset_dir / "val" / "images"
    val_labels_dir = dataset_dir / "val" / "labels"

    # Create directories if they don't exist
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = sorted(list(images_dir.glob("*")))
    print(f"Total images: {len(image_files)}")

    # Shuffle for random split
    random.seed(seed)
    random.shuffle(image_files)

    # Calculate split point
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    print(f"Train set: {len(train_files)} images ({train_ratio*100:.0f}%)")
    print(f"Val set: {len(val_files)} images ({(1-train_ratio)*100:.0f}%)")

    # Copy files to train
    for img_file in train_files:
        # Copy image
        shutil.copy2(img_file, train_images_dir / img_file.name)

        # Copy label
        label_file = labels_dir / (img_file.stem + ".txt")
        if label_file.exists():
            shutil.copy2(label_file, train_labels_dir / label_file.name)

    # Copy files to val
    for img_file in val_files:
        # Copy image
        shutil.copy2(img_file, val_images_dir / img_file.name)

        # Copy label
        label_file = labels_dir / (img_file.stem + ".txt")
        if label_file.exists():
            shutil.copy2(label_file, val_labels_dir / label_file.name)

    print("\n✅ Dataset split complete!")
    print(f"Train: {train_images_dir}")
    print(f"Val: {val_images_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset into train/val sets for YOLO training"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to dataset directory (must contain images/ and labels/ folders)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of training data (default: 0.8 for 80/20 split)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Validate dataset directory
    if not args.dataset_dir.exists():
        print(f"❌ Error: Dataset directory not found: {args.dataset_dir}")
        exit(1)

    if not (args.dataset_dir / "images").exists():
        print(f"❌ Error: images/ folder not found in {args.dataset_dir}")
        exit(1)

    if not (args.dataset_dir / "labels").exists():
        print(f"❌ Error: labels/ folder not found in {args.dataset_dir}")
        exit(1)

    # Run split
    split_dataset(args.dataset_dir, args.train_ratio, args.seed)
