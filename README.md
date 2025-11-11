# Road Anomaly Detection using YOLOv8

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)

Real-time road damage detection system using YOLOv8 on the RDD2022 dataset for identifying and classifying four types of road anomalies: longitudinal cracks, transverse cracks, alligator cracks, and potholes. This project includes GPS integration and severity classification for practical deployment in road maintenance systems.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Results](#results)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Database Schema](#database-schema)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Dataset

### RDD2022 (Road Damage Dataset 2022)
- **Source**: [RDD2022 Official Dataset](https://rdd2022.sekilab.global/)
- **Classes**: 4 road damage types
  - D00: Longitudinal Crack
  - D10: Transverse Crack
  - D20: Alligator Crack
  - D40: Pothole
- **Image Resolution**: 640×640 pixels
- **Split**: Train/Val/Test
- **Format**: YOLO format annotations

For detailed dataset information, see [data/README.md](data/README.md)

## Installation

### Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU training)
- 8GB+ GPU memory (recommended)
- 20GB+ disk space

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/road-anomaly-detection.git
cd road-anomaly-detection

# Install dependencies
pip install -r requirements.txt

# Download dataset (if not included)
# Place RDD2022 dataset in data/ directory
```
## Model Weights

Due to GitHub's 25MB file size limit, the trained model weights are hosted externally.

**Download:** [YOLOv8 Best Weights (best.pt)](https://drive.google.com/file/d/1aS8aMj1G33Cs3G9YPAfpda2VikRMTxhq/view?usp=sharing)

### Quick Setup:
```bash
#Install gdown
gdown https://drive.google.com/uc?id=1aS8aMj1G33Cs3G9YPAfpda2VikRMTxhq -O models/weights/best.pt
gdown https://drive.google.com/uc?id=1Q_nWO8bW0HHb7X8AT3FiP7zutBgYxD4N -O models/weights/last.pt
```

## Model Architecture

**Model**: YOLOv8-Medium (yolov8m.pt)

### Key Features
- **Backbone**: CSPDarknet with C2f modules
- **Neck**: PAN-FPN architecture
- **Head**: Decoupled detection head
- **Parameters**: ~25M
- **Input Size**: 640×640
- **Output**: Bounding boxes with class predictions and confidence scores

### Why YOLOv8-Medium?
- Optimal balance between accuracy and inference speed
- Suitable for real-time road damage detection
- Better feature extraction for small objects (cracks)
- Efficient on edge devices

## Training Details

### Hyperparameters
```yaml
Epochs: 20 (recommended: 100-150 for production)
Batch Size: 8
Image Size: 640×640
Device: GPU (CUDA)
Optimizer: AdamW
Learning Rate (initial): 0.0001
Learning Rate (final): 0.0001
Momentum: 0.9
Weight Decay: 0.0001
Warmup Epochs: 5
```

### Data Augmentation
```yaml
Mosaic: 0.8
Mixup: 0.0
Horizontal Flip: 0.3
Vertical Flip: 0.3
Rotation: ±5°
Translation: ±5%
Scale: ±30%
HSV Augmentation: H(0.01), S(0.3), V(0.2)
```

### Loss Configuration
```yaml
Box Loss Weight: 5.0
Class Loss Weight: 0.3
DFL Loss Weight: 1.0
```

### Training Command
```bash
python src/train.py --data data/rdd2022.yaml --epochs 20 --batch 8 --device 0
```

## Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.XXXX |
| mAP@0.5:0.95 | 0.XXXX |
| Precision | 0.XXXX |
| Recall | 0.XXXX |
| F1-Score | 0.XXXX |

*Note: Replace with actual values after training*

### Per-Class Performance

| Class | Precision | Recall | mAP@0.5 |
|-------|-----------|--------|---------|
| D00 Longitudinal Crack | 0.XXX | 0.XXX | 0.XXX |
| D10 Transverse Crack | 0.XXX | 0.XXX | 0.XXX |
| D20 Alligator Crack | 0.XXX | 0.XXX | 0.XXX |
| D40 Pothole | 0.XXX | 0.XXX | 0.XXX |

### Sample Detections
See results/sample_detections/ for detection examples

### Training Curves
See results/plots/ for training metrics visualization

## Usage

### Training
```bash
# Train from scratch
python src/train.py --data data/rdd2022.yaml --epochs 20 --batch 8

# Resume training
python src/train.py --data data/rdd2022.yaml --resume --weights models/weights/last.pt
```

### Validation
```bash
# Validate trained model
python src/validate.py --weights models/weights/best.pt --data data/rdd2022.yaml
```

### Inference

#### Single Image
```bash
python src/inference.py --weights models/weights/best.pt --source test_images/image.jpg
```

#### Batch Inference
```bash
python src/inference.py --weights models/weights/best.pt --source test_images/ --save-db
```

#### Real-time Video
```bash
python src/inference.py --weights models/weights/best.pt --source video.mp4 --save-db
```

### Using the Trained Model in Python
```python
from ultralytics import YOLO

# Load model
model = YOLO('models/weights/best.pt')

# Run inference
results = model('path/to/image.jpg', conf=0.3)

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        print(f"Class: {class_id}, Conf: {confidence:.2f}, BBox: {bbox}")
```

## Project Structure

```
road-anomaly-detection/
├── README.md                  # Main documentation
├── LICENSE                    # MIT License
├── requirements.txt           # Python dependencies
├── data/
│   ├── rdd2022.yaml          # YOLO dataset configuration
│   └── README.md             # Dataset documentation
├── src/
│   ├── train.py              # Training script
│   ├── validate.py           # Validation script
│   ├── inference.py          # Inference with database logging
│   └── utils.py              # Helper functions
├── models/
│   └── weights/              # Trained model weights
│       ├── best.pt           # Best model checkpoint
│       └── last.pt           # Last epoch checkpoint
├── configs/
│   └── training_config.yaml  # Training configuration
├── notebooks/
│   └── exploratory_analysis.ipynb  # Data exploration
├── results/
│   ├── metrics.json          # Performance metrics
│   ├── plots/                # Training curves and visualizations
│   └── sample_detections/    # Sample detection images
└── docs/
    └── setup_instructions.md # Detailed setup guide
```

## Database Schema

The inference pipeline stores detections in SQLite database for tracking and analysis.

### Detections Table
```sql
CREATE TABLE detections (
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
);
```

### Severity Classification
- **High**: Bounding box area > 50,000 pixels²
- **Medium**: 20,000 < area ≤ 50,000 pixels²
- **Low**: area ≤ 20,000 pixels²

## Future Improvements

### Model Enhancements
- Implement attention mechanisms (LSKA, DAT)
- Use MPDIoU loss for better localization
- Add C2fGhost modules for efficiency
- Experiment with YOLOv8x for higher accuracy
- Knowledge distillation from larger models

### Training Optimizations
- Extend training to 100-150 epochs
- Implement class-specific confidence thresholds
- Add focal loss for handling class imbalance
- Use learning rate scheduler (cosine annealing)
- Increase batch size with gradient accumulation

### System Features
- Real-time GPS integration with mapping API
- Multi-model ensemble for improved accuracy
- Automatic severity assessment using damage area
- Web dashboard for detection visualization
- Mobile app for field deployment
- Integration with municipal maintenance systems

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **RDD2022 Dataset**: Sekilab, University of Tokyo
- **YOLOv8**: Ultralytics team
- **CICPS 2026 Hackathon**: JUSense organizers

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{road_anomaly_detection_2025,
  title={Road Anomaly Detection using YOLOv8},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/road-anomaly-detection}
}
```

## Contact

For questions or collaborations:
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)

---
**CICPS 2026 Hackathon Submission - Team [Your Team Name]**
