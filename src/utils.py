#!/usr/bin/env python3
"""
Road Anomaly Detection - Utility Functions
Helper functions for data processing, visualization, and analysis
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sqlite3
import pandas as pd

# Class names and colors
CLASS_NAMES = {
    0: 'D00_Longitudinal_Crack',
    1: 'D10_Transverse_Crack',
    2: 'D20_Alligator_Crack',
    3: 'D40_Pothole'
}

CLASS_COLORS = {
    0: (255, 0, 0),      # Blue for D00
    1: (0, 255, 0),      # Green for D10
    2: (0, 165, 255),    # Orange for D20
    3: (0, 0, 255)       # Red for D40
}

def load_image(image_path):
    """Load and return an image"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return img

def save_image(image, output_path):
    """Save image to disk"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)

def draw_bbox(image, bbox, class_id, confidence, label=True):
    """Draw bounding box on image"""
    x1, y1, x2, y2 = map(int, bbox)
    color = CLASS_COLORS.get(class_id, (255, 255, 255))

    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Draw label
    if label:
        class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
        label_text = f"{class_name}: {confidence:.2f}"

        # Get label size
        (label_width, label_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Draw label background
        cv2.rectangle(image, (x1, y1 - label_height - baseline - 5), 
                     (x1 + label_width, y1), color, -1)

        # Draw label text
        cv2.putText(image, label_text, (x1, y1 - baseline - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image

def visualize_detections(image_path, detections, output_path=None, show=False):
    """Visualize detections on image

    Args:
        image_path: Path to input image
        detections: List of dicts with keys: bbox, class_id, confidence
        output_path: Path to save visualization
        show: Whether to display image
    """
    img = load_image(image_path)

    for det in detections:
        img = draw_bbox(img, det['bbox'], det['class_id'], det['confidence'])

    if output_path:
        save_image(img, output_path)

    if show:
        cv2.imshow('Detections', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection area
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height

    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def plot_training_curves(results_csv, output_dir='results/plots'):
    """Plot training curves from results CSV"""
    os.makedirs(output_dir, exist_ok=True)

    # Read results
    df = pd.read_csv(results_csv)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)

    # Plot losses
    if 'train/box_loss' in df.columns:
        axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Box Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

    if 'train/cls_loss' in df.columns:
        axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Class Loss', color='orange')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Classification Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

    # Plot mAP
    if 'metrics/mAP50(B)' in df.columns:
        axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mAP')
        axes[1, 0].set_title('mAP@0.5')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # Plot Precision/Recall
    if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
        axes[1, 1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='blue')
        axes[1, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Precision & Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training curves saved: {output_path}")
    plt.close()

def create_confusion_matrix_plot(confusion_matrix, class_names, output_path):
    """Create confusion matrix visualization"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion_matrix, cmap='Blues')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, f'{confusion_matrix[i, j]:.2f}',
                          ha='center', va='center', color='black')

    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {output_path}")
    plt.close()

def export_database_to_csv(db_path, output_path='results/detections.csv'):
    """Export database detections to CSV"""
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM detections", conn)
    conn.close()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Database exported to CSV: {output_path}")

    return df

def analyze_detections(db_path):
    """Analyze detections from database"""
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)

    # Summary statistics
    print("\n" + "="*70)
    print("DETECTION ANALYSIS")
    print("="*70)

    # Total detections
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM detections")
    total = c.fetchone()[0]
    print(f"\nTotal detections: {total}")

    # By class
    print("\nDetections by damage type:")
    c.execute("SELECT class_name, COUNT(*) as count FROM detections GROUP BY class_name ORDER BY count DESC")
    for row in c.fetchall():
        print(f"  {row[0]}: {row[1]} ({row[1]/total*100:.1f}%)")

    # By severity
    print("\nDetections by severity:")
    c.execute("SELECT severity, COUNT(*) as count FROM detections GROUP BY severity ORDER BY count DESC")
    for row in c.fetchall():
        print(f"  {row[0]}: {row[1]} ({row[1]/total*100:.1f}%)")

    # Average confidence
    print("\nAverage confidence by class:")
    c.execute("SELECT class_name, AVG(confidence) as avg_conf FROM detections GROUP BY class_name")
    for row in c.fetchall():
        print(f"  {row[0]}: {row[1]:.3f}")

    conn.close()
    print("="*70)

def create_detection_heatmap(db_path, output_path='results/detection_heatmap.png'):
    """Create heatmap of detection locations (if GPS data available)"""
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT latitude, longitude, severity FROM detections", conn)
    conn.close()

    # Filter out zero coordinates
    df = df[(df['latitude'] != 0) & (df['longitude'] != 0)]

    if len(df) == 0:
        print("No GPS data available for heatmap")
        return

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))

    severity_colors = {'High': 'red', 'Medium': 'orange', 'Low': 'yellow'}
    for severity, color in severity_colors.items():
        data = df[df['severity'] == severity]
        ax.scatter(data['longitude'], data['latitude'], 
                  c=color, label=severity, alpha=0.6, s=50)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Detection Locations by Severity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Detection heatmap saved: {output_path}")
    plt.close()

def validate_dataset_structure(data_dir):
    """Validate YOLO dataset structure"""
    print("\n" + "="*70)
    print("DATASET VALIDATION")
    print("="*70)

    data_path = Path(data_dir)

    # Check directories
    required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
    for dir_name in required_dirs:
        dir_path = data_path / dir_name
        if dir_path.exists():
            count = len(list(dir_path.glob('*')))
            print(f"✓ {dir_name}: {count} files")
        else:
            print(f"✗ {dir_name}: NOT FOUND")

    print("="*70)

if __name__ == "__main__":
    # Example usage
    print("Utility functions loaded successfully")
    print("\nAvailable functions:")
    print("  - visualize_detections()")
    print("  - plot_training_curves()")
    print("  - export_database_to_csv()")
    print("  - analyze_detections()")
    print("  - create_detection_heatmap()")
    print("  - validate_dataset_structure()")
