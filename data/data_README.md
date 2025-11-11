# RDD2022 Dataset Documentation

## Overview
The Road Damage Dataset 2022 (RDD2022) is a large-scale dataset for road damage detection containing images from multiple countries with diverse road conditions.

## Dataset Information

### Source
- **Official Website**: https://rdd2022.sekilab.global/
- **Citation**: Sekilab, University of Tokyo
- **Year**: 2022
- **License**: Check official website for usage terms

### Classes (4 types)
1. **D00 - Longitudinal Crack**: Linear cracks parallel to road direction
2. **D10 - Transverse Crack**: Linear cracks perpendicular to road direction
3. **D20 - Alligator Crack**: Interconnected cracks forming patterns
4. **D40 - Pothole**: Bowl-shaped depressions in road surface

## Dataset Statistics

### Image Information
- **Total Images**: [Add your count]
- **Image Format**: JPG/PNG
- **Image Size**: Variable (resized to 640×640 for training)
- **Color Space**: RGB

### Data Split
```
Train:      [XX]% ([XXXX] images)
Validation: [XX]% ([XXXX] images)
Test:       [XX]% ([XXXX] images)
```

### Class Distribution
| Class | Train | Val | Test | Total |
|-------|-------|-----|------|-------|
| D00 Longitudinal Crack | XXX | XX | XX | XXX |
| D10 Transverse Crack | XXX | XX | XX | XXX |
| D20 Alligator Crack | XXX | XX | XX | XXX |
| D40 Pothole | XXX | XX | XX | XXX |

## Dataset Structure

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

## Annotation Format

### YOLO Format
Each `.txt` file contains bounding box annotations:
```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `class_id`: 0-3 (D00, D10, D20, D40)
- `x_center, y_center`: Normalized center coordinates (0-1)
- `width, height`: Normalized box dimensions (0-1)

Example:
```
0 0.512 0.345 0.123 0.089
3 0.678 0.456 0.234 0.178
```

## Preprocessing Steps

1. **Image Resizing**: All images resized to 640×640 pixels
2. **Normalization**: Pixel values normalized to [0, 1]
3. **Augmentation**: Applied during training (see main README)

## Data Quality

### Image Quality Criteria
- Minimum resolution: 640×480
- Clear visibility of road surface
- Proper lighting conditions
- No significant occlusions

### Annotation Quality
- Tight bounding boxes around damage
- Accurate class labels
- Multiple annotators for verification

## Download Instructions

### Option 1: Official Website
```bash
# Download from official RDD2022 website
# Follow instructions at: https://rdd2022.sekilab.global/
```

### Option 2: Kaggle (if available)
```bash
# Download using Kaggle API
kaggle datasets download -d [dataset-name]
unzip [dataset-name].zip -d data/
```

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{rdd2022,
  title={Road Damage Detection and Classification Challenge},
  author={Arya, Deeksha and Maeda, Hiroya and Ghosh, Sanjay Kumar and others},
  booktitle={IEEE BigData},
  year={2022}
}
```

## Data Augmentation Used

### Training Augmentations
- Mosaic (0.8 probability)
- Horizontal flip (0.3)
- Vertical flip (0.3)
- Random rotation (±5°)
- Random translation (±5%)
- Random scale (±30%)
- HSV augmentation

### Validation/Test
- No augmentation (only resize)

## Known Issues and Limitations

1. **Class Imbalance**: Some damage types more frequent than others
2. **Lighting Variation**: Images captured under different lighting conditions
3. **Scale Variation**: Damage sizes vary significantly
4. **Background Complexity**: Urban vs rural road backgrounds

## Recommendations

- Use weighted loss to handle class imbalance
- Apply confidence threshold tuning per class
- Consider ensemble methods for better accuracy
- Validate on diverse road conditions

---
Last Updated: November 2025
