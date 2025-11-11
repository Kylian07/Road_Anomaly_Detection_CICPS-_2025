#!/usr/bin/env python3
"""
Road Anomaly Detection - Inference Script
Run detection on images/videos with GPS logging and database storage
"""

import os
import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import cv2
import json

# Class names mapping
CLASS_NAMES = {
    0: 'D00_Longitudinal_Crack',
    1: 'D10_Transverse_Crack',
    2: 'D20_Alligator_Crack',
    3: 'D40_Pothole'
}

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Road Damage Detection Inference')
    parser.add_argument('--weights', type=str, default='models/weights/best.pt',
                        help='Path to model weights')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image/video/directory')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for inference')
    parser.add_argument('--save-db', action='store_true',
                        help='Save detections to database')
    parser.add_argument('--db-path', type=str, default='results/road_detections.db',
                        help='Path to SQLite database')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save results to TXT files')
    parser.add_argument('--save-img', action='store_true', default=True,
                        help='Save annotated images')
    parser.add_argument('--output', type=str, default='results/inference',
                        help='Output directory')
    parser.add_argument('--gps-file', type=str, default=None,
                        help='JSON file with GPS coordinates (format: {filename: [lat, lon]})')
    return parser.parse_args()

def create_detection_db(db_path):
    """Create SQLite database for detections"""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT,
        class_id INTEGER,
        class_name TEXT,
        confidence REAL,
        x_min REAL,
        y_min REAL,
        x_max REAL,
        y_max REAL,
        bbox_area REAL,
        latitude REAL,
        longitude REAL,
        timestamp TEXT,
        severity TEXT
    )''')

    conn.commit()
    return conn

def calculate_severity(bbox_area):
    """Calculate damage severity based on bounding box area"""
    if bbox_area > 50000:
        return 'High'
    elif bbox_area > 20000:
        return 'Medium'
    else:
        return 'Low'

def load_gps_data(gps_file):
    """Load GPS coordinates from JSON file"""
    if gps_file and os.path.exists(gps_file):
        with open(gps_file, 'r') as f:
            return json.load(f)
    return {}

def run_inference(args):
    """Run inference on images/videos"""
    print("="*70)
    print("ROAD ANOMALY DETECTION - INFERENCE")
    print("="*70)
    print(f"\nModel: {args.weights}")
    print(f"Source: {args.source}")
    print(f"Confidence: {args.conf}")
    print(f"IoU: {args.iou}")

    # Check if model exists
    if not os.path.exists(args.weights):
        print(f"\nError: Model weights not found at {args.weights}")
        return

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load model
    print("\nLoading model...")
    model = YOLO(args.weights)

    # Initialize database if needed
    conn = None
    if args.save_db:
        conn = create_detection_db(args.db_path)
        c = conn.cursor()
        print(f"✓ Database initialized: {args.db_path}")

    # Load GPS data if provided
    gps_data = load_gps_data(args.gps_file)
    if gps_data:
        print(f"✓ GPS data loaded: {len(gps_data)} entries")

    # Get list of images
    source_path = Path(args.source)
    if source_path.is_file():
        image_files = [source_path]
    else:
        image_files = list(source_path.glob('*.jpg')) + list(source_path.glob('*.png')) + list(source_path.glob('*.jpeg'))

    print(f"\nProcessing {len(image_files)} images...\n")

    # Run inference
    detection_count = 0
    for idx, img_path in enumerate(image_files):
        img_name = img_path.name

        try:
            # Run detection
            results = model(str(img_path), conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)

            for result in results:
                # Save annotated image
                if args.save_img:
                    output_path = os.path.join(args.output, img_name)
                    result.save(output_path)

                # Process detections
                if len(result.boxes) > 0:
                    # Get GPS coordinates
                    lat, lon = gps_data.get(img_name, [0.0, 0.0])

                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = CLASS_NAMES[class_id]

                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        bbox_area = (x2 - x1) * (y2 - y1)
                        severity = calculate_severity(bbox_area)

                        # Save to database
                        if args.save_db and conn:
                            c.execute('''INSERT INTO detections 
                                (image_name, class_id, class_name, confidence, x_min, y_min, 
                                 x_max, y_max, bbox_area, latitude, longitude, timestamp, severity)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                (img_name, class_id, class_name, confidence,
                                 x1, y1, x2, y2, bbox_area, lat, lon, 
                                 datetime.now().isoformat(), severity))

                        # Save to TXT file
                        if args.save_txt:
                            txt_path = os.path.join(args.output, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
                            with open(txt_path, 'a') as f:
                                f.write(f"{class_id} {confidence:.4f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")

                        detection_count += 1

            # Progress update
            if (idx + 1) % 10 == 0:
                print(f"✓ Processed {idx + 1}/{len(image_files)} images")

        except Exception as e:
            print(f"✗ Error processing {img_name}: {str(e)}")

    # Commit database changes
    if conn:
        conn.commit()
        conn.close()

    print(f"\n" + "="*70)
    print(f"INFERENCE COMPLETE")
    print("="*70)
    print(f"\nTotal detections: {detection_count}")
    print(f"Output directory: {args.output}")
    if args.save_db:
        print(f"Database: {args.db_path}")

def query_database(db_path, query_type='all'):
    """Query detection database"""
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    if query_type == 'all':
        c.execute("SELECT * FROM detections LIMIT 10")
        print("\nFirst 10 detections:")
        for row in c.fetchall():
            print(row)

    elif query_type == 'summary':
        c.execute("SELECT class_name, COUNT(*) as count FROM detections GROUP BY class_name")
        print("\nDetections by class:")
        for row in c.fetchall():
            print(f"  {row[0]}: {row[1]}")

        c.execute("SELECT severity, COUNT(*) as count FROM detections GROUP BY severity")
        print("\nDetections by severity:")
        for row in c.fetchall():
            print(f"  {row[0]}: {row[1]}")

    conn.close()

def main():
    """Main inference pipeline"""
    args = parse_args()
    run_inference(args)

    # Print database summary if database was used
    if args.save_db:
        print("\n")
        query_database(args.db_path, query_type='summary')

if __name__ == "__main__":
    main()
