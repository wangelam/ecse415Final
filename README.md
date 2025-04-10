# ECSE 415 Final Project: Tumor Detection and Segmentation

![Medical Imaging](https://img.shields.io/badge/domain-medical_imaging-blue) ![Machine Learning](https://img.shields.io/badge/ML-supervised%20%7C%20unsupervised-orange) ![Python](https://img.shields.io/badge/python-3.8%2B-blue)

## 📂 Complete File Listing

### Core Project Files
- `README.md` - This documentation file
- `requirements.txt` - Python dependencies
- `.DS_Store` - System file (can be ignored)

### 3. Dataset Preprocessing
- `preprocess.ipynb` - Handles data splitting and all preprocessing steps

### 4. Segmentation Algorithms
#### 4.1 Unsupervised Approaches
- `unsupervised.ipynb` - Contains K-means and Mean Shift implementations
- `utils.py` - Shared helper functions for feature extraction and utilities

#### 4.2 Supervised Approaches
- `Supervised_RandomForest.ipynb` - Random Forest classifier implementation
- `Supervised_SVM.ipynb` - Support Vector Machine classifier

### 5. Tumor Detection System
- `TumorCount_ROISVM.ipyn` - Final optimized version with performance metrics
- `tumor.ipynb` - Additional SVM experiments on original images

### 6. Deep Learning (Bonus)
- `multiclass_seg_finetune.ipynb` - U-Net++ fine-tuning for multiclass segmentation



## 🚀 Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
pip install segmentation-models-pytorch
