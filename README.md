# Weather Classification Model (Rainy vs. Clear)

A deep learning model that classifies weather conditions as **rainy** or **clear** using TensorFlow/Keras with MobileNetV2. Achieves **96% training accuracy** and **90% validation accuracy**, with generalization tested on real-world videos.

![accuracy_plot](accuracy_plot.png) 

## Features
- **Binary Classification**: Predicts "rainy" (🌧️) or "clear" (☀️) weather from images.
- **Model**: Fine-tuned `MobileNetV2` (transfer learning) with class balancing.
- **Dataset**: 
  - Rainy images: `leftImg8bit_trainval_rain` (Cityscapes subset) + custom videos for generalization.
  - Clear images: `leftImg8bit_trainvaltest` + custom videos for generalization.
- **Inference**: Filters and labels images/videos with confidence scores.

## Code Structure
```
weather-classification/
├── dataset/                  # Training data
│   ├── train/                # Training images (rainy/clear subfolders)
│   └── val/                  # Validation images
├── dataset2/                 # Test images/videos
├── rain_model_training.py    # Model training script
├── rain_model_inference.py   # Prediction and visualization
├── best_model.h5             # Saved model (best weights)
├── training_history.json     # Training metrics
└── accuracy_plot.png         # Accuracy/loss plot (auto-generated)
```

## Installation
1. **Clone the repo**:
   ```bash
   git clone https://github.com/ghaidaasamir/Weather-Classification-Model-Rainy-vs.-Clear-.git
   cd weather-classification
   ```

2. **Install dependencies**:
   ```bash
   pip install tensorflow opencv-python scikit-learn
   ```

## Usage
### 1. Training
```bash
python rain_model_training.py
```
- **Input**: Images in `dataset/train/` and `dataset/val/` (subfolders: `rainy`, `clear`).
- **Output**: 
  - `best_model.h5` (best weights during training).
  - `training_history.json` (metrics log).

### 2. Inference
```bash
python rain_model_inference.py
```
- **Input**: Images in `dataset2/clear_and_rainy_drops_images/`.
- **Output**: 
  - Labeled images in `clear_and_rainy_drops_images_check_folder/`.
  - Predictions log: `clear_and_rainy_drops_images.txt`.

## Results
| Metric          | Score |
|-----------------|-------|
| Training Accuracy | 96%   |
| Validation Accuracy | 90%   |
| Tested on Videos | Robust performance |

**Demo Video**: [Download Link](https://drive.google.com/file/d/1H8SxN5ZMp7Rq83mAG2NNME7JAPMwaee0/view?usp=sharing) 

## Customization
- **Threshold Adjustment**: Modify `threshold=0.5` in `rain_model_inference.py` for stricter/looser predictions.
- **Model**: Replace `MobileNetV2` in `rain_model_training.py` (e.g., with ResNet).

---
