#!/usr/bin/env python3
"""
Watermark Cleaning Script using LaMa
Reads a detection report (JSON) and cleans watermarks from images using LaMa inpainting.
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Add LaMa environment path to sys.path if needed
# This assumes the script is run from the project root or with the correct environment
try:
    from iopaint.model.lama import LaMa
    from iopaint.schema import InpaintRequest, HDStrategy
    from iopaint.helper import pad_img_to_modulo
except ImportError:
    print("Error: iopaint module not found. Please ensure you are running in the 'lama_env' environment.")
    sys.exit(1)

def load_report(report_path):
    """Load detection report from JSON file"""
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_mask_from_bbox(image_shape, bboxes, padding=10):
    """
    Create a binary mask from bounding boxes
    
    Args:
        image_shape: (height, width) of the image
        bboxes: List of [x1, y1, x2, y2] coordinates
        padding: Extra padding around the bounding box
        
    Returns:
        Binary mask (numpy array) where 255 is the area to inpaint
    """
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        
        # Draw filled rectangle on mask
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
    return mask

# Global model for multiprocessing workers
worker_model = None

def init_worker(device):
    global worker_model
    # Force CPU for worker if device is cpu
    if device == 'cpu':
        torch.set_num_threads(1) # Avoid oversubscription
    worker_model = LaMa(device)
    worker_model.init_model(device)

def process_image_worker(args):
    """Worker function for processing a single image"""
    item, output_dir, padding, resize_limit = args
    global worker_model
    
    try:
        image_path = item['filepath']
        filename = item['filename']
        detections = item['detections']
        
        if not os.path.exists(image_path):
            return False, f"Warning: Image not found: {image_path}"
            
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return False, f"Warning: Could not read image: {image_path}"
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        height, width = img_rgb.shape[:2]

        max_dim = resize_limit
        scale = 1.0
        if max_dim > 0 and max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_rgb = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
            height, width = new_height, new_width
            detections = [{'bbox': [c * scale for c in det['bbox']]} for det in detections]

        img_rgb = pad_img_to_modulo(img_rgb, 8)
        height, width = img_rgb.shape[:2]
        
        bboxes = [det['bbox'] for det in detections]
        mask = create_mask_from_bbox((height, width), bboxes, padding)
        
        config = InpaintRequest(
            ldm_steps=20,
            ldm_sampler='plms',
            hd_strategy=HDStrategy.ORIGINAL,
            hd_strategy_crop_margin=128,
            hd_strategy_crop_trigger_size=2048,
            hd_strategy_resize_limit=2048,
        )
        
        with torch.no_grad():
            result_bgr = worker_model.forward(img_rgb, mask, config)
        
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, result_bgr)
        
        return True, None
    except Exception as e:
        return False, f"Error processing {filename}: {str(e)}"

def clean_images(report_path, output_dir, device='cuda', padding=10, resize_limit=4096, control_file=None, workers=1):
    """
    Clean watermarks from images listed in the report
    """
    # Load report
    print(f"Loading report from: {report_path}")
    report = load_report(report_path)
    
    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get images with watermarks
    images_to_process = report.get('results', {}).get('with_watermark', [])
    
    if not images_to_process:
        print("No images with watermarks found in the report.")
        return
    
    print(f"Found {len(images_to_process)} images to process.")
    
    success_count = 0
    error_count = 0

    # CPU Multiprocessing Mode
    if device == 'cpu' and workers > 1:
        print(f"Starting CPU multiprocessing with {workers} workers...")
        
        # Prepare args for each item
        process_args = [(item, output_dir, padding, resize_limit) for item in images_to_process]
        
        # Use spawn context for safety with PyTorch
        ctx = multiprocessing.get_context('spawn')
        
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=init_worker, initargs=(device,)) as executor:
            futures = []
            for args in process_args:
                # Check control file
                if control_file and os.path.exists(control_file):
                    try:
                        with open(control_file, 'r') as f:
                            control_data = json.load(f)
                            status = control_data.get('status', 'running')
                            if status == 'stopped':
                                print("\nProcess stopped by user.")
                                executor.shutdown(wait=False, cancel_futures=True)
                                return
                            while status == 'paused':
                                print("\rProcess paused. Waiting...", end="", flush=True)
                                time.sleep(1)
                                with open(control_file, 'r') as f2:
                                    control_data = json.load(f2)
                                    status = control_data.get('status', 'running')
                                    if status == 'stopped':
                                        print("\nProcess stopped by user.")
                                        executor.shutdown(wait=False, cancel_futures=True)
                                        return
                    except Exception:
                        pass
                
                futures.append(executor.submit(process_image_worker, args))
            
            for future in tqdm(futures, desc="Cleaning images (CPU Multi)"):
                try:
                    success, error = future.result()
                    if success:
                        success_count += 1
                    else:
                        print(error)
                        error_count += 1
                except Exception as e:
                    print(f"Worker error: {e}")
                    error_count += 1
                    
    else:
        # GPU or Single Process Mode
        print(f"Initializing LaMa model on {device}...")
        model = LaMa(device)
        model.init_model(device)
        
        for item in tqdm(images_to_process, desc="Cleaning images"):
            # Check control file
            if control_file and os.path.exists(control_file):
                try:
                    with open(control_file, 'r') as f:
                        control_data = json.load(f)
                        status = control_data.get('status', 'running')
                        
                        if status == 'stopped':
                            print("\nProcess stopped by user.")
                            return
                        
                        while status == 'paused':
                            print("\rProcess paused. Waiting...", end="", flush=True)
                            time.sleep(1)
                            with open(control_file, 'r') as f2:
                                control_data = json.load(f2)
                                status = control_data.get('status', 'running')
                                if status == 'stopped':
                                    print("\nProcess stopped by user.")
                                    return
                except Exception as e:
                    print(f"Error reading control file: {e}")

            try:
                image_path = item['filepath']
                filename = item['filename']
                detections = item['detections']
                
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    error_count += 1
                    continue
                    
                # Load image
                img_bgr = cv2.imread(image_path)
                if img_bgr is None:
                    print(f"Warning: Could not read image: {image_path}")
                    error_count += 1
                    continue
                    
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                height, width = img_rgb.shape[:2]

                # Resize
                max_dim = resize_limit
                scale = 1.0
                if max_dim > 0 and max(height, width) > max_dim:
                    scale = max_dim / max(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img_rgb = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    height, width = new_height, new_width
                    detections = [{'bbox': [c * scale for c in det['bbox']]} for det in detections]

                # Pad
                img_rgb = pad_img_to_modulo(img_rgb, 8)
                height, width = img_rgb.shape[:2]
                
                # Mask
                bboxes = [det['bbox'] for det in detections]
                mask = create_mask_from_bbox((height, width), bboxes, padding)
                
                config = InpaintRequest(
                    ldm_steps=20,
                    ldm_sampler='plms',
                    hd_strategy=HDStrategy.ORIGINAL,
                    hd_strategy_crop_margin=128,
                    hd_strategy_crop_trigger_size=2048,
                    hd_strategy_resize_limit=2048,
                )
                
                try:
                    with torch.no_grad():
                        result_bgr = model.forward(img_rgb, mask, config)
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) and device == 'cuda':
                        print(f"OOM for {filename}, switching to CPU for this image...")
                        torch.cuda.empty_cache()
                        original_device = model.device
                        model.model.to('cpu')
                        model.device = torch.device('cpu')
                        try:
                            with torch.no_grad():
                                result_bgr = model.forward(img_rgb, mask, config)
                        finally:
                            model.model.to('cuda')
                            model.device = original_device
                    else:
                        raise e
                
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, result_bgr)
                
                success_count += 1

                if device == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                error_count += 1
            
    print("\nProcessing complete!")
    print(f"Successfully cleaned: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Cleaned images saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Clean watermarks from images using LaMa and detection report")
    parser.add_argument("--report", "-r", required=True, help="Path to the JSON detection report")
    parser.add_argument("--output", "-o", required=True, help="Directory to save cleaned images")
    parser.add_argument("--device", "-d", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--padding", "-p", type=int, default=10, help="Padding around detection bounding boxes (pixels)")
    parser.add_argument("--resize-limit", type=int, default=4096, help="Maximum dimension for resizing images (0 to disable)")
    parser.add_argument("--control-file", help="Path to control file for pause/stop functionality")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers for CPU multiprocessing")
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, switching to CPU")
        args.device = 'cpu'
        
    clean_images(args.report, args.output, args.device, args.padding, args.resize_limit, args.control_file, args.workers)

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()