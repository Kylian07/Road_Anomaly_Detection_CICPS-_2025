# Setup Instructions - Road Anomaly Detection

Complete guide for setting up and running the Road Anomaly Detection system using YOLOv8 on the RDD2022 dataset.

## Prerequisites

### System Requirements
- **Operating System**: Ubuntu 20.04+ / Windows 10+ / macOS 11+
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended for training)
- **CUDA**: 11.8 or higher (for GPU acceleration)
- **RAM**: 16GB+ recommended
- **Disk Space**: 20GB+ for dataset and models

### Software Requirements
- Git
- Python 3.8+
- pip or conda package manager
- CUDA Toolkit (for GPU training)

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/road-anomaly-detection.git
cd road-anomaly-detection
```

### 2. Create Virtual Environment

**Using venv (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n road-detection python=3.8
conda activate road-detection
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Check YOLOv8 installation
python -c "from ultralytics import YOLO; print('YOLOv8 installed successfully')"
```

## Dataset Setup

### Option 1: Download RDD2022 Dataset

1. Visit the official website: https://rdd2022.sekilab.global/
2. Register and download the dataset
3. Extract the dataset to the `data/` directory

### Option 2: Use Kaggle Dataset

```bash
# Install Kaggle API
pip install kaggle

# Configure Kaggle credentials
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset (replace with actual dataset name)
kaggle datasets download -d [rdd2022-dataset-name]
unzip [dataset-name].zip -d data/
```

### Dataset Structure

Ensure your dataset follows this structure:
```
data/
├── rdd2022.yaml          # YOLO configuration file
├── train/
│   ├── images/           # Training images
│   └── labels/           # Training annotations (YOLO format)
├── val/
│   ├── images/           # Validation images
│   └── labels/           # Validation annotations
└── test/
    ├── images/           # Test images
    └── labels/           # Test annotations (optional)
```

### Validate Dataset

```bash
python src/utils.py
# Or run validation function
python -c "from src.utils import validate_dataset_structure; validate_dataset_structure('data')"
```

## Model Weights

Due to GitHub's 25MB file size limit, the trained model weights are hosted externally.

**Download:** [YOLOv8 Best Weights (best.pt)](https://drive.google.com/file/d/1aS8aMj1G33Cs3G9YPAfpda2VikRMTxhq/view?usp=sharing)
**Download:** [YOLOv8 last Weights (last.pt)](https://drive.google.com/file/d/1Q_nWO8bW0HHb7X8AT3FiP7zutBgYxD4N/view?usp=sharing)

### Quick Setup:
```bash
#Install gdown
pip install gdown

#Download model weights
gdown https://drive.google.com/uc?id=1aS8aMj1G33Cs3G9YPAfpda2VikRMTxhq -O models/weights/best.pt
gdown https://drive.google.com/uc?id=1Q_nWO8bW0HHb7X8AT3FiP7zutBgYxD4N -O models/weights/last.pt

#Or download manually from the link above and place in models/weights/
```

### Verify Model Weights

```bash
ls -lh models/weights/best.pt
# Should show file size around 50MB
```

## Kaggle Setup (for Kaggle Notebooks)

### 1. Create New Notebook

- Go to Kaggle.com
- Click "Code" → "New Notebook"
- Enable GPU: Settings → Accelerator → GPU T4 x2

### 2. Clone Repository

```python
# In Kaggle notebook
!git clone https://github.com/yourusername/road-anomaly-detection.git
%cd road-anomaly-detection
```

### 3. Install Dependencies

```python
!pip install -q ultralytics
```

### 4. Add Dataset

- Click "Add Data" → Search for RDD2022
- Or upload your dataset as a Kaggle dataset

### 5. Download Model Weights

```python
!pip install gdown
!gdown YOUR_FILE_ID -O models/weights/best.pt
```

## Google Colab Setup

### 1. Open Colab Notebook

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/yourusername/road-anomaly-detection.git
%cd road-anomaly-detection

# Install dependencies
!pip install -r requirements.txt
```

### 2. Enable GPU

- Runtime → Change runtime type → GPU (T4 or A100)

### 3. Download Model and Dataset

```python
# Download model weights
!pip install gdown
!gdown YOUR_FILE_ID -O models/weights/best.pt

# Upload dataset or link from Google Drive
```

## Training

### Quick Start Training

```bash
python src/train.py
```

### Custom Training Parameters

```bash
python src/train.py \
  --data data/rdd2022.yaml \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --device 0
```

### Training with Configuration File

```bash
# Edit configs/training_config.yaml with your preferences
python src/train.py --config configs/training_config.yaml
```

### Monitor Training

Training logs and plots will be saved to:
- Output directory: `output/yolov8m_rdd2022/`
- Training plots: `output/yolov8m_rdd2022/results.png`
- Model weights: `output/yolov8m_rdd2022/weights/best.pt`

**TensorBoard (optional):**
```bash
tensorboard --logdir output/yolov8m_rdd2022/
```

## Validation

### Run Validation on Trained Model

```bash
python src/validate.py \
  --weights models/weights/best.pt \
  --data data/rdd2022.yaml \
  --split val
```

### Custom Validation Parameters

```bash
python src/validate.py \
  --weights models/weights/best.pt \
  --data data/rdd2022.yaml \
  --split test \
  --batch 16 \
  --conf 0.3 \
  --save-json \
  --plots
```

### View Validation Results

- Metrics JSON: `results/metrics.json`
- Metrics CSV: `results/metrics_table_overall.csv`
- Per-class CSV: `results/metrics_table_per_class.csv`
- Confusion matrix: `output/yolov8m_rdd2022/confusion_matrix.png`

## Inference

### Single Image Inference

```bash
python src/inference.py \
  --weights models/weights/best.pt \
  --source test_images/image.jpg \
  --save-img
```

### Batch Image Processing

```bash
python src/inference.py \
  --weights models/weights/best.pt \
  --source test_images/ \
  --save-img \
  --save-db \
  --db-path results/road_detections.db
```

### Video Inference

```bash
python src/inference.py \
  --weights models/weights/best.pt \
  --source video.mp4 \
  --save-img
```

### Inference with GPS Data

```bash
# Create GPS JSON file: {"image1.jpg": [lat, lon], "image2.jpg": [lat, lon]}
python src/inference.py \
  --weights models/weights/best.pt \
  --source test_images/ \
  --save-db \
  --gps-file gps_coordinates.json
```

### View Inference Results

- Annotated images: `results/inference/`
- Detection database: `results/road_detections.db`
- Detection CSV: Export using `src/utils.py`

## Using Python API

### Training

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8m.pt')

# Train
results = model.train(
    data='data/rdd2022.yaml',
    epochs=20,
    batch=8,
    imgsz=640
)
```

### Inference

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('models/weights/best.pt')

# Run inference
results = model('test_images/image.jpg', conf=0.3)

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        print(f"Class: {class_id}, Confidence: {confidence:.2f}")
```

## Common Issues and Solutions

### Issue 1: CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
- Reduce batch size: `--batch 4` or `--batch 2`
- Reduce image size: `--imgsz 480`
- Use CPU (slower): `--device cpu`
- Close other GPU applications

### Issue 2: Dataset Not Found

**Error:** `FileNotFoundError: Dataset not found`

**Solutions:**
- Check paths in `data/rdd2022.yaml`
- Ensure images and labels are in correct directories
- Verify file permissions: `ls -la data/train/images/`
- Run dataset validation: `python -c "from src.utils import validate_dataset_structure; validate_dataset_structure('data')"`

### Issue 3: Module Not Found

**Error:** `ModuleNotFoundError: No module named 'ultralytics'`

**Solutions:**
- Activate virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

### Issue 4: Slow Training

**Performance Issues:**

**Solutions:**
- Enable image caching: Set `cache=True` in training config
- Reduce workers if CPU limited: `--workers 2`
- Use smaller model: `yolov8n.pt` or `yolov8s.pt`
- Upgrade GPU or use cloud GPU (Kaggle/Colab)

### Issue 5: Poor Detection Results

**Low mAP or accuracy:**

**Solutions:**
- Train for more epochs (100-150 recommended)
- Increase learning rate: `lr0=0.001`
- Check data quality and annotations
- Try different confidence thresholds: `--conf 0.2` or `--conf 0.4`
- Use data augmentation (already enabled)

### Issue 6: Model File Too Large for GitHub

**Error:** File size exceeds 25MB

**Solutions:**
- Use Git LFS: `git lfs track "*.pt"`
- Upload to Google Drive and share link
- Use Hugging Face Hub for model hosting
- See `LARGE_FILES_GUIDE.md` for detailed instructions

## Performance Optimization

### Training Optimization

- **Mixed Precision Training**: Enabled by default with `amp=True`
- **Increase Batch Size**: If GPU memory allows, use `--batch 16` or `--batch 32`
- **Image Caching**: Enable with `cache=True` for faster data loading
- **Multi-GPU Training**: Use `--device 0,1` for multiple GPUs

### Inference Optimization

- **Batch Processing**: Process multiple images together
- **Model Variants**: Use smaller models (yolov8n, yolov8s) for faster inference
- **TensorRT**: Export to TensorRT for production deployment
- **ONNX Export**: For cross-platform deployment

## Model Export

### Export to ONNX

```python
from ultralytics import YOLO

model = YOLO('models/weights/best.pt')
model.export(format='onnx')
```

### Export to TensorRT

```python
model.export(format='engine', device=0)
```

### Export to CoreML (for iOS)

```python
model.export(format='coreml')
```

## Database Operations

### Query Detection Database

```python
import sqlite3

conn = sqlite3.connect('results/road_detections.db')
cursor = conn.cursor()

# Get all high-severity potholes
cursor.execute("""
    SELECT * FROM detections 
    WHERE class_name='D40_Pothole' AND severity='High'
    ORDER BY confidence DESC
""")

results = cursor.fetchall()
conn.close()
```

### Export Database to CSV

```python
from src.utils import export_database_to_csv

export_database_to_csv(
    'results/road_detections.db',
    'results/detections_export.csv'
)
```

### Analyze Detections

```python
from src.utils import analyze_detections

analyze_detections('results/road_detections.db')
```

## Visualization

### Plot Training Curves

```python
from src.utils import plot_training_curves

plot_training_curves(
    'output/yolov8m_rdd2022/results.csv',
    'results/plots/'
)
```

### Create Detection Heatmap

```python
from src.utils import create_detection_heatmap

create_detection_heatmap(
    'results/road_detections.db',
    'results/detection_heatmap.png'
)
```

## For Hackathon Evaluators

### Quick Evaluation Setup

```bash
# 1. Clone repository
git clone https://github.com/yourusername/road-anomaly-detection.git
cd road-anomaly-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download model weights
pip install gdown
gdown YOUR_FILE_ID -O models/weights/best.pt

# 4. Run validation
python src/validate.py --weights models/weights/best.pt --data data/rdd2022.yaml

# 5. Run inference on sample images
python src/inference.py --weights models/weights/best.pt --source test_images/
```

### Expected Results

- **mAP@0.5**: ~0.XX-0.XX (replace with your actual metrics)
- **Inference Speed**: ~XX ms per image on GPU
- **Model Size**: ~50MB

## Additional Resources

- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **RDD2022 Dataset**: https://rdd2022.sekilab.global/
- **PyTorch Documentation**: https://pytorch.org/docs/
- **Project Repository**: [Your GitHub Link]

## Contact and Support

For questions or issues:
- **GitHub Issues**: [repository-url]/issues
- **Email**: your.email@example.com
- **CICPS 2026 Hackathon**: jusense.org

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

**Last Updated**: November 2025  
**CICPS 2026 Hackathon Submission**
