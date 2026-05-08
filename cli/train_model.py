#!/usr/bin/env python3
"""
Fine-tune YOLOv8 watermark detection model
"""

import argparse
from ultralytics import YOLO
from pathlib import Path


def train_model(
    base_model: Path,
    dataset_yaml: Path,
    output_dir: Path,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    patience: int = 10,
    device: str = "0",
    name: str = "watermark_finetune"
):
    """
    Fine-tune YOLOv8 model

    Args:
        base_model: Path to base/pretrained model (.pt file)
        dataset_yaml: Path to dataset YAML configuration
        output_dir: Directory to save training results
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        patience: Early stopping patience
        device: Device to use (0 for GPU, cpu for CPU)
        name: Experiment name
    """
    print("=" * 50)
    print("YOLOv8 Watermark Detection - Fine-tuning")
    print("=" * 50)
    print(f"Base model: {base_model}")
    print(f"Dataset: {dataset_yaml}")
    print(f"Output: {output_dir}")
    print()

    # Load pre-trained model
    print("Loading pre-trained model...")
    model = YOLO(str(base_model))

    # Train the model
    print("\nStarting training...")
    print(f"- Epochs: {epochs}")
    print(f"- Image size: {imgsz}")
    print(f"- Batch size: {batch}")
    print(f"- Patience: {patience}")
    print(f"- Device: {device}")
    print(f"- Name: {name}")
    print()

    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name,
        project=str(output_dir),
        exist_ok=True,
        patience=patience,
        save=True,
        device=device,
        verbose=True,
    )

    print("\n" + "=" * 50)
    print("✅ Training Complete!")
    print("=" * 50)
    print(f"\nBest model saved at:")
    print(f"{output_dir / name / 'weights' / 'best.pt'}")
    print(f"\nLast model saved at:")
    print(f"{output_dir / name / 'weights' / 'last.pt'}")
    print(f"\nResults:")
    print(f"{output_dir / name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 watermark detection model"
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        required=True,
        help="Path to base/pretrained model (.pt file)"
    )
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        required=True,
        help="Path to dataset YAML configuration"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save training results"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training (default: 640)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use: 0 for first GPU, cpu for CPU (default: 0)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="watermark_finetune",
        help="Experiment name (default: watermark_finetune)"
    )

    args = parser.parse_args()

    # Validate paths
    if not args.base_model.exists():
        print(f"❌ Error: Base model not found: {args.base_model}")
        exit(1)

    if not args.dataset_yaml.exists():
        print(f"❌ Error: Dataset YAML not found: {args.dataset_yaml}")
        exit(1)

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run training
    train_model(
        args.base_model,
        args.dataset_yaml,
        args.output_dir,
        args.epochs,
        args.imgsz,
        args.batch,
        args.patience,
        args.device,
        args.name
    )
