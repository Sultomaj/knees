# Knee Osteoarthritis Diagnostic Classifier (EfficientNet-V2)

## System Overview
This repository contains the architecture and training pipeline for a medical diagnostic model designed to classify the severity of Knee Osteoarthritis from radiographic images. Built using PyTorch, the system leverages a fine-tuned `EfficientNet-V2-S` backbone to provide high-accuracy, resource-efficient inference suitable for clinical deployment.

### 🔬 Clinical & Research Highlights
While raw accuracy is a standard metric in machine learning, this pipeline was explicitly engineered with **clinical safety** and **physiological reality** in mind.

* **Prioritizing Patient Safety (Minimizing False Negatives):** In medical diagnostics, the cost of a false negative far outweighs a false positive. Through iterative tuning, this model was optimized to aggressively catch severe Osteoporosis. We intentionally traded a fraction of overall accuracy to drastically reduce the critical risk of missing advanced joint deterioration.
* **Physiologically Sound Augmentations:** The data pipeline strictly avoids destructive spatial transforms (such as heavy random rotations) that artificially distort the horizontal joint space. Instead, it relies on structure-preserving augmentations to ensure the network learns true clinical biomarkers rather than orientation artifacts.
* **Transparent Evaluation & Next Steps:** The included confusion matrices were generated to evaluate ordinal progression errors. The data clearly isolates the model's remaining blind spots, providing a precise roadmap for future research iterations (such as implementing ordinal-weighted loss functions to heavily penalize multi-stage misclassifications).

## My Core Engineering Contributions
* **Clinical Data Imbalance Mitigation:** Engineered a weighted loss function using inverse frequency class weighting (`sklearn.utils.class_weight`) to prevent the model from mirroring the inherent biases of the medical dataset.
* **Test Time Augmentation (TTA) Pipeline:** Implemented a custom TTA pipeline during validation to ensure robust, stable predictions across varying radiological image qualities, a critical requirement for deploying models across different hospital hardware.
* **Pipeline Optimization:** Utilized `StepLR` scheduling and optimized PyTorch `DataLoaders` for efficient batch processing. Separated training logic and metric evaluation into modular components for high reproducibility.

## Tech Stack
* **Framework:** PyTorch, Torchvision
* **Architecture:** EfficientNet-V2-S (Transfer Learning)
* **Data Processing:** Scikit-learn, NumPy, Pandas
* **Visualization:** Matplotlib
* **Data Pipeline:** Kagglehub, tqdm

## Model Evaluation & Performance
The model was rigorously evaluated to ensure high recall across minority classes, specifically addressing the overlapping visual features between intermediate osteoarthritis severity levels. 

*(Note: Replace this text with the image of your Matplotlib Loss/Accuracy curves)*
`![Learning Curves](results/learning_curves.png)`

*(Note: Replace this text with the image of your Matplotlib `show_predictions` grid)*
`![Prediction Samples](results/prediction_samples.png)`

*(Note: Replace this text with the image of your Confusion Matrix)*
`![Confusion Matrix](results/confusion_matrix.png)`

## How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
2. **Download the Dataset:**
   ```bash
   import kagglehub
kagglehub.dataset_download("fuyadhasanbhoyan/knee-osteoarthritis-classification-224224")
3. **Train the Model:**
   ``` bash
    python3 train.py
