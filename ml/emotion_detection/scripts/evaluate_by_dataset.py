"""
Evaluate the already-trained combined model separately on
RAF-DB test and AffectNet test, without retraining.
"""
import sys
import torch
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))
from ml.emotion_detection.model.emotion_cnn import EmotionCNN
from ml.emotion_detection.data.dataset_loader import raf_test_loader, affectnet_test_loader
from ml.emotion_detection.config import (BEST_MODEL_PATH, EMOTION_NAMES, NUM_CLASSES)
from ml.emotion_detection.utils import get_device, plot_confusion_matrix

# --- Evaluate Model ---
def evaluate_by_dataset(model, loader, name, device):
    correct = 0     # Number of correctly classified images.
    total = 0    # Total number of images evaluated.
    # class_correct[i] stores how many images from class i were correctly classified.
    class_correct = [0] * NUM_CLASSES
    #class_total[i] stores how many test images actually belong to class i.
    class_total = [0] * NUM_CLASSES
    # These lists will store every true label and prediction for the confusion matrix 
    all_labels = []
    all_predictions = []

    # no_grad() bc evaluating
    with torch.no_grad():
        # Iterate through every batch in the test DataLoader.
        for images, labels in loader:
            # batch of input images and there labels
            images = images.to(device)
            labels = labels.to(device)
            # Logits 
            outputs = model(images)
            # torch.max(outputs, 1) looks across the seven class outputs for each image.
            # get the value (which we through away) and more importantly the index/emotion ID for that highest predicted label 
            _, predicted = torch.max(outputs, 1)
            # Add the number of images in this batch to the total num of evaluated images
            total += labels.size(0)
            # Counts how many predictions were correct and turn Pytorch val into python num with .item()
            correct += (predicted == labels).sum().item()
            # Move the labels back to the CPU and convert them to NumPy arrays, add every label from batch to list
            all_labels.extend(labels.cpu().numpy())
            # Model's predictions.
            all_predictions.extend(predicted.cpu().numpy())

            # --- Per Class Accuracy --- 
            # Go through each true label and its corresponding prediction in the current batch.
            # pair them together with zip: true label  <-> prediction
            for label, pred in zip(labels, predicted):
                # Actual Emotion ID  
                label = label.item()
                # Predicted Emotion ID
                pred = pred.item()
                class_total[label] += 1
                # If correct, iterate correct count
                if label == pred:
                    class_correct[label] += 1
    # --- Overall Accuracy --- 
    accuracy = 100 * correct / total
    print(f"\n{'=' * 60}")
    print(f"{name} — Overall Accuracy: {accuracy:.2f}%")
    print(f"{'=' * 60}")
    # --- Per Class Results --- 
    for i, emotion in enumerate(EMOTION_NAMES):
        if class_total[i] > 0:
            #  Calculate the percentage of examples from this that were correctly classified
            class_acc = 100 * class_correct[i] / class_total[i]
        else:
            class_acc = 0.0
        print(f"{emotion:<12} {class_correct[i]:>4} / {class_total[i]:<4} ({class_acc:>6.2f}%)")

    # Return the complete true-label and prediction lists
    return all_labels, all_predictions

# --- Main ---
def main():
    # --- Select Device ---
    device = get_device()
    print(f"Using device: {device}")

    # --- Model ---
    model = EmotionCNN()
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval() # Eval Mode.

    # --- Evaluate RAF-DB and AffectNet ---
    raf_labels, raf_preds = evaluate_by_dataset(
        model,
        raf_test_loader,
        "RAF-DB TEST ONLY",
        device
    )

    affectnet_labels, affectnet_preds = evaluate_by_dataset(
        model,
        affectnet_test_loader,
        "AFFECTNET TEST ONLY",
        device
    )

    # --- Confusion Matrix for RAF-DB ---
    plot_confusion_matrix(
        raf_labels,
        raf_preds,
        "RAF-DB Test Confusion Matrix (Combined-Trained Model)"
    )
    # --- Confusion Matrix for AffectNet ---
    plot_confusion_matrix(
        affectnet_labels,
        affectnet_preds,
        "AffectNet Test Confusion Matrix (Combined-Trained Model)"
    )

# --- Driver --- 
if __name__ == "__main__":
    main()