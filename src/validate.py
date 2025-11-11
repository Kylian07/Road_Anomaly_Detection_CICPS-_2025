#!/usr/bin/env python3
"""
Road Anomaly Detection - Validation Script
Evaluate YOLOv8 model on validation/test set
"""

import os
import json
import argparse
from pathlib import Path
from ultralytics import YOLO
import pandas as pd

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Validate YOLOv8 Road Damage Detection Model')
    parser.add_argument('--weights', type=str, default='models/weights/best.pt',
                        help='Path to model weights')
    parser.add_argument('--data', type=str, default='data/rdd2022.yaml',
                        help='Path to dataset YAML file')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                        help='Dataset split to validate on')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size for validation')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for validation')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='IoU threshold for NMS')
    parser.add_argument('--save-json', action='store_true',
                        help='Save results to JSON file')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save results to TXT files')
    parser.add_argument('--plots', action='store_true', default=True,
                        help='Generate confusion matrix and other plots')
    return parser.parse_args()

def validate_model(args):
    """Run validation on the model"""
    print("="*70)
    print("ROAD ANOMALY DETECTION - VALIDATION")
    print("="*70)
    print(f"\nModel: {args.weights}")
    print(f"Dataset: {args.data}")
    print(f"Split: {args.split}")
    print(f"Batch Size: {args.batch}")
    print(f"Image Size: {args.imgsz}")
    print(f"Confidence: {args.conf}")
    print(f"IoU: {args.iou}")

    # Check if model exists
    if not os.path.exists(args.weights):
        print(f"\nError: Model weights not found at {args.weights}")
        return None

    # Load model
    print("\nLoading model...")
    model = YOLO(args.weights)

    # Run validation
    print("\nRunning validation...")
    results = model.val(
        data=args.data,
        split=args.split,
        batch=args.batch,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        save_json=args.save_json,
        save_txt=args.save_txt,
        plots=args.plots,
        verbose=True
    )

    return results

def print_metrics(results):
    """Print validation metrics"""
    print("\n" + "="*70)
    print("VALIDATION METRICS")
    print("="*70)

    # Overall metrics
    print("\nOverall Performance:")
    print(f"  mAP@0.5:      {results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"  Precision:    {results.box.mp:.4f}")
    print(f"  Recall:       {results.box.mr:.4f}")

    # Calculate F1-Score
    f1 = 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-6)
    print(f"  F1-Score:     {f1:.4f}")

    # Per-class metrics
    print("\nPer-Class Performance:")
    class_names = ['D00_Longitudinal_Crack', 'D10_Transverse_Crack', 
                   'D20_Alligator_Crack', 'D40_Pothole']

    if hasattr(results.box, 'ap_class_index') and hasattr(results.box, 'ap'):
        for idx, class_name in enumerate(class_names):
            if idx < len(results.box.ap_class_index):
                print(f"\n  {class_name}:")
                print(f"    Precision: {results.box.class_result(idx)[0]:.4f}")
                print(f"    Recall:    {results.box.class_result(idx)[1]:.4f}")
                print(f"    mAP@0.5:   {results.box.class_result(idx)[2]:.4f}")

    print("\n" + "="*70)

def save_metrics_json(results, output_path='results/metrics.json'):
    """Save metrics to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Calculate F1-Score
    f1 = 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-6)

    metrics_dict = {
        'overall': {
            'mAP@0.5': float(results.box.map50),
            'mAP@0.5:0.95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'f1_score': float(f1)
        }
    }

    # Add per-class metrics if available
    class_names = ['D00_Longitudinal_Crack', 'D10_Transverse_Crack', 
                   'D20_Alligator_Crack', 'D40_Pothole']

    metrics_dict['per_class'] = {}
    if hasattr(results.box, 'ap_class_index'):
        for idx, class_name in enumerate(class_names):
            if idx < len(results.box.ap_class_index):
                try:
                    class_metrics = results.box.class_result(idx)
                    metrics_dict['per_class'][class_name] = {
                        'precision': float(class_metrics[0]),
                        'recall': float(class_metrics[1]),
                        'mAP@0.5': float(class_metrics[2])
                    }
                except:
                    pass

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"\n✓ Metrics saved to: {output_path}")

    return metrics_dict

def create_metrics_table(metrics_dict, output_path='results/metrics_table.csv'):
    """Create CSV table of metrics"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Overall metrics
    df_overall = pd.DataFrame([metrics_dict['overall']])
    df_overall.to_csv(output_path.replace('.csv', '_overall.csv'), index=False)

    # Per-class metrics
    if 'per_class' in metrics_dict and metrics_dict['per_class']:
        df_class = pd.DataFrame(metrics_dict['per_class']).T
        df_class.to_csv(output_path.replace('.csv', '_per_class.csv'))
        print(f"✓ Metrics tables saved to: {os.path.dirname(output_path)}")

def main():
    """Main validation pipeline"""
    args = parse_args()

    # Run validation
    results = validate_model(args)

    if results is not None:
        # Print metrics
        print_metrics(results)

        # Save metrics
        metrics_dict = save_metrics_json(results)
        create_metrics_table(metrics_dict)

        print("\n✓ Validation complete!")
    else:
        print("\n✗ Validation failed!")

if __name__ == "__main__":
    main()
