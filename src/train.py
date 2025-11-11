#!/usr/bin/env python3
"""
Road Anomaly Detection - Training Script
YOLOv8 training on RDD2022 dataset
"""

import os
import json
from pathlib import Path
from ultralytics import YOLO
import torch

# Configuration
DATASET_ROOT = '/kaggle/working/rdd2022_yolo'
OUTPUT_PATH = '/kaggle/working/output'
YAML_FILE = os.path.join(OUTPUT_PATH, 'rdd2022.yaml')

# Training parameters
EPOCHS = 20
BATCH_SIZE = 8
IMAGE_SIZE = 640
DEVICE = 0  # GPU device (0 for first GPU, 'cpu' for CPU)

def setup_directories():
    """Create necessary directories"""
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, 'weights'), exist_ok=True)
    print(f"✓ Output directory: {OUTPUT_PATH}")

def create_yaml_config():
    """Create YOLO dataset configuration"""
    yaml_content = f"""path: {DATASET_ROOT}
train: train/images
val: val/images
test: test/images

nc: 4
names: ['D00_Longitudinal_Crack', 'D10_Transverse_Crack', 'D20_Alligator_Crack', 'D40_Pothole']
"""

    with open(YAML_FILE, 'w') as f:
        f.write(yaml_content)

    print(f"✓ YAML config created: {YAML_FILE}")

def train_model():
    """Train YOLOv8 model"""
    print("\n" + "="*70)
    print("TRAINING YOLOV8")
    print("="*70)

    # Load pretrained model
    model = YOLO('yolov8m.pt')
    print(f"\n✓ YOLOv8 Medium loaded")
    print(f"✓ Starting training...\n")

    # Train
    results = model.train(
        data=YAML_FILE,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        patience=30,
        save=True,
        project=OUTPUT_PATH,
        name='yolov8m_rdd2022',
        cache=False,
        workers=4,

        # Augmentation (optimized)
        augment=True,
        mosaic=0.8,
        mixup=0.0,
        flipud=0.3,
        fliplr=0.3,
        degrees=5,
        translate=0.05,
        scale=0.3,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,

        # Optimizer (better convergence)
        optimizer='AdamW',
        lr0=0.0001,
        lrf=0.0001,
        momentum=0.9,
        weight_decay=0.0001,
        warmup_epochs=5,
        warmup_momentum=0.9,
        warmup_bias_lr=0.001,

        # Loss weights
        box=5.0,
        cls=0.3,
        dfl=1.0,

        # Inference
        conf=0.3,
        iou=0.5,

        verbose=True,
        plots=True,
    )

    print("\n✓ Training complete!")
    return results

def validate_model():
    """Validate trained model"""
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)

    best_model_path = os.path.join(OUTPUT_PATH, 'yolov8m_rdd2022/weights/best.pt')

    if os.path.exists(best_model_path):
        model = YOLO(best_model_path)
        print("\nValidating...")
        metrics = model.val()

        print("\n" + "="*70)
        print("PERFORMANCE METRICS")
        print("="*70)

        metrics_dict = {
            'mAP@0.5': float(metrics.box.map50),
            'mAP@0.5:0.95': float(metrics.box.map),
            'Precision': float(metrics.box.mp),
            'Recall': float(metrics.box.mr),
        }

        f1 = 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-6)
        metrics_dict['F1-Score'] = float(f1)

        print(f"\n✓ mAP@0.5:      {metrics.box.map50:.4f}")
        print(f"✓ mAP@0.5:0.95: {metrics.box.map:.4f}")
        print(f"✓ Precision:    {metrics.box.mp:.4f}")
        print(f"✓ Recall:       {metrics.box.mr:.4f}")
        print(f"✓ F1-Score:     {f1:.4f}")

        # Save metrics
        metrics_json = os.path.join(OUTPUT_PATH, 'metrics.json')
        with open(metrics_json, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        print(f"\n✓ Metrics saved: {metrics_json}")
        return metrics_dict
    else:
        print(f"Error: Model not found at {best_model_path}")
        return None

def main():
    """Main training pipeline"""
    print("="*70)
    print("ROAD ANOMALY DETECTION - TRAINING PIPELINE")
    print("="*70)

    # Setup
    setup_directories()
    create_yaml_config()

    # Train
    train_model()

    # Validate
    metrics = validate_model()

    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
