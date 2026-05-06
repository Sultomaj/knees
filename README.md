# Knee Osteoarthritis Diagnostic Classifier (EfficientNet-V2)

## System Overview
This repository contains the architecture and training pipeline for a medical diagnostic model designed to classify the severity of Knee Osteoarthritis from radiographic images. Built using PyTorch, the system leverages a fine-tuned `EfficientNet-V2-S` backbone to provide high-accuracy, resource-efficient inference suitable for clinical deployment.

## My Core Engineering Contributions
*   **Clinical Data Imbalance Mitigation:** Engineered a weighted loss function using inverse frequency class weighting (`sklearn.utils.class_weight`) to prevent the model from mirroring the inherent biases of the medical dataset.
*   **Medical-Safe Data Augmentation:** Designed a custom augmentation pipeline tailored for radiological imagery. Replaced destructive standard transforms (like heavy rotation, which skews joint space narrowing) with medical-safe augmentations (horizontal flips and subtle affine translations) to ensure the model learns true physiological markers rather than orientation artifacts.
*   **Pipeline Optimization:** Utilized `StepLR` scheduling and optimized PyTorch `DataLoaders` for efficient batch processing. Separated training logic and metric evaluation into modular components for high reproducibility.

## Tech Stack
*   **Framework:** PyTorch, Torchvision
*   **Architecture:** EfficientNet-V2-S (Transfer Learning)
*   **Data Processing:** Scikit-learn, NumPy, Pandas
*   **Visualization:** Matplotlib
*   **Data Pipeline:** Kagglehub, tqdm

## Model Evaluation & Performance
The model was rigorously evaluated to ensure high recall across minority classes, specifically addressing the overlapping visual features between intermediate osteoarthritis severity levels. 

**Final Metrics:**
*   **Overall Test Accuracy:** 84.07%
*   **Class 0 (Normal):** 75.83%
*   **Class 1 (Osteopenia):** 82.50%
*   **Class 2 (Osteoporosis):** 77.78%

### Learning Curves
*(Note: Replace this text with the image of your Matplotlib Loss/Accuracy curves)*
`![Learning Curves](results/learning_curves.png)`

### Prediction Samples
*(Note: Replace this text with the image of your Matplotlib `show_predictions` grid)*
`![Prediction Samples](results/prediction_samples.png)`

### Confusion Matrix
*(Note: Replace this text with the image of your Confusion Matrix)*
`![Confusion Matrix](results/confusion_matrix.png)`

## How to Run

1. **Install Dependencies:**
   Ensure you have Python 3.8+ installed, then install the required packages:
   ```bash
   pip install torch torchvision scikit-learn matplotlib pandas numpy kagglehub tqdm
