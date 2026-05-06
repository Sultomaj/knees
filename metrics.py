import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import json
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def validate(model, dataloader, criterion, device):
    """Standard validation loop."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def test_model(model, test_loader, device):
    """Evaluates the model on the test dataset."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def show_predictions(model, dataloader, device, class_names, num_images=10, save_path=None):
    """Displays and saves a grid of images with their predicted and true labels."""
    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat[:num_images]):
        if i >= len(images):
            break
            
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = img * std + mean
        img = np.clip(img, 0, 1)

        pred_label = predictions[i].item()
        true_label = labels[i].item()

        if pred_label >= len(class_names) or true_label >= len(class_names):
            continue

        ax.imshow(img)
        ax.set_title(f"Pred: {class_names[pred_label]}\nTrue: {class_names[true_label]}")
        ax.axis("off")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction image saved to {save_path}")
    
    plt.show()
    plt.close()

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs, save_path=None):
    """Plots and saves training and validation loss and accuracy curves."""
    epochs_range = range(1, num_epochs + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss", marker='o')
    plt.plot(epochs_range, val_losses, label="Validation Loss", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label="Train Accuracy", marker='o')
    plt.plot(epochs_range, val_accuracies, label="Validation Accuracy", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics plot saved to {save_path}")
        
    plt.show()
    plt.close()

    
def plot_confusion_matrix(model, dataloader, device, class_names, save_path=None):
    """Generates and saves a confusion matrix to visualize class confusion."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Store predictions and true labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute the matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot it
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    
    plt.title("Test Set Confusion Matrix")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        
    plt.show()
    plt.close()

def per_class_accuracy(model, val_loader, device, num_classes=3):
    """Calculates accuracy for each individual class and returns it as a dictionary."""
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    results = {}

    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(num_classes):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            results[f"Class_{i}"] = round(acc, 2)
    
    return results

def save_metrics_to_file(metrics_dict, save_path):
    """Saves numerical metrics to a JSON file for easy reading."""
    with open(save_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Numerical metrics saved to {save_path}")

def validate_with_tta(model, dataloader, criterion, device, n_tta=5):
    """Performs validation using Test Time Augmentation (TTA)."""
    tta_transforms = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2),
    ])

    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            tta_preds = []
            for _ in range(n_tta):
                augmented_images = torch.stack([tta_transforms(img) for img in images])
                outputs = model(augmented_images)
                tta_preds.append(outputs)

            avg_preds = torch.mean(torch.stack(tta_preds), dim=0)
            loss = criterion(avg_preds, labels)
            total_loss += loss.item()

            _, predicted = torch.max(avg_preds, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy, torch.cat(all_preds), torch.cat(all_labels)