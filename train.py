import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
import kagglehub
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm

# Import local metric functions
from metrics import (
    validate, 
    test_model, 
    show_predictions, 
    plot_metrics, 
    per_class_accuracy, 
    validate_with_tta,
    save_metrics_to_file,
    plot_confusion_matrix
)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Trains the model for a single epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def main():
    # Setup Results Directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # 1. Setup Environment and Download Data
    os.environ['KAGGLEHUB_CACHE'] = "./data"
    path = kagglehub.dataset_download("fuyadhasanbhoyan/knee-osteoarthritis-classification-224224")
    print(f"Dataset path: {path}")

    # 2. Define Transformations
    transform_train = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize((224, 224)),
        #transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),           # Safe: Left vs Right knee
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_valid_test = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 3. Initialize Datasets and DataLoaders
    train_path = os.path.join(path, "Knee Osteoarthritis Classification", "train")
    test_path = os.path.join(path, "Knee Osteoarthritis Classification", "test")
    val_path = os.path.join(path, "Knee Osteoarthritis Classification", "val")

    train_dataset = ImageFolder(train_path, transform=transform_train)
    test_dataset = ImageFolder(test_path, transform=transform_valid_test)
    val_dataset = ImageFolder(val_path, transform=transform_valid_test)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    class_names = train_dataset.classes
    print(f"Classes ({len(class_names)}): {class_names}")

    # 4. Model Setup (EfficientNet_V2_S)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, len(class_names))
    )
    model = model.to(device)

    # 5. Class Weights and Optimization
    targets = train_loader.dataset.targets
    class_labels = np.unique(targets)
    class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=targets)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 6. Training Loop
    num_epochs = 10
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    print("Starting Training...")
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}, Train Acc={train_accuracy:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.2f}%")
        
        scheduler.step()

    # Save model weights
    model_save_path = os.path.join(results_dir, "knee_osteoarthritis_efficientnet.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # 7. Evaluation and Saving Results
    print("\nEvaluating on Test Set...")
    final_test_acc = test_model(model, test_loader, device)
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")

    print("\nCalculating Per-Class Accuracy...")
    class_acc_dict = per_class_accuracy(model, val_loader, device, num_classes=len(class_names))
    for cls, acc in class_acc_dict.items():
        print(f"{cls}: {acc}%")

    # Compile all text metrics into a dictionary
    final_metrics = {
        "final_test_accuracy_percent": round(final_test_acc, 2),
        "best_val_accuracy_percent": round(max(val_accuracies), 2),
        "per_class_accuracy_percent": class_acc_dict
    }

    # Save text metrics to JSON
    save_metrics_to_file(final_metrics, os.path.join(results_dir, "final_metrics.json"))

    print("\nGenerating and Saving Visualizations...")
    # Save plots
    plot_metrics(
        train_losses, val_losses, train_accuracies, val_accuracies, num_epochs, 
        save_path=os.path.join(results_dir, "learning_curves.png")
    )
    
    show_predictions(
        model, test_loader, device, class_names, 
        save_path=os.path.join(results_dir, "prediction_samples.png")
    )
    plot_confusion_matrix(
        model, test_loader, device, class_names, 
        save_path=os.path.join(results_dir, "confusion_matrix.png")
    )

if __name__ == "__main__":
    main()