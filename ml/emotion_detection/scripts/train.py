"""
Training script for the facial emotion recognition CNN.

This file connects the different parts of the machine learning pipeline:
Dataset -> DataLoader -> CNN model -> Forward pass -> Loss calculation -> Backpropagation -> Optimizer -> Updated model weights

The trained model is then evaluated on the test dataset and saved
to disk so that it can later be used for emotion prediction.
"""

import torch
import torch.nn as nn # NN building blocks 
import torch.optim as optim # For optimization functions 
import torch
from ml.emotion_detection.model.emotion_cnn import EmotionCNN
from data.dataset_loader import train_loader, val_loader, test_loader
from ml.emotion_detection.config import (EMOTION_NAMES, EPOCHS, LEARNING_RATE, NUM_CLASSES)
from ml.emotion_detection.utils import (get_device, plot_confusion_matrix)

# --- Training Configuration ---- 
from config import EMOTION_NAMES, EPOCHS, LEARNING_RATE, NUM_CLASSES
# NUM_CLASSES = 7, One logit / each emotion class 
# LEARNING_RATE = 0.001, How large the optimizer's parameter updates -> how much to update weights by -> weight' = weight - learning_rate * gradient 
# EPOCHS = 50, Num of complete passes through training set
EXPERIMENT_MODEL_PATH = ("/Users/laurenpalega/Documents/virtualfriend/ml/emotion_detection/checkpoints/emotion_cnn_raf_affectnet_experiment.pth")

def train_model(model, criterion, optimizer, device):
    """
    Train the CNN and save the best model based on validation accuracy.
    """
    # --- Training Loop ---- 
    # So each batch of images goes through 4 processes 
    # -> pass in images, compute loss, backpropgate to get gradients, then update parameters (+ update stats)
    best_val_accuracy = 0.0 
    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0 # Accumulate loss across epochs to calculate avg loss
        correct = 0 # Num of correctly classified images 
        total = 0 # Num of classifed images 
        total_batches = len(train_loader)
        # for images, labels in train_loader:
        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad() # PyTorch accumulates gradients by default, so each batch we clear it

            # FOWARD PASS
            # CNN outputs a 2D tensor  -> 32 images × 7 emotion logits
            outputs = model(images) # Send images through CNN -> forward() in emotion_cnn.py 

            # CALCULATE LOSS 
            loss = criterion(outputs, labels) # How wrong predicitons are with current weights and baises 

            # BACKPROPAGATION  
            # Pytorch calculates differentiation to calculate gradients of loss w.r.t each trainable parameter
            # Goal: calculate gradient that points in direction that reduces that loss -> we need to tweak those weights and biases by to reduce error
            loss.backward()   

            # UPDATE PARAMS
            # Actually update weight and loss params 
            # Gradient calculates which direction to reduce loss, learning rate defines step 
            optimizer.step() 

            # UPDATE LOSS STAT
            running_loss += loss.item() # coverts tensor containing loss to a num

            # CALCULATE PREDICITONS
            # Look across dimension 1 of tensor (the 7 emotion classes) 
            # 7 logits for each emotion -> largest logic represents predicted class
            # Using _ for largest logit bc we really only need the predicted index -> aka corresponding class label
            # OUTPUT: largest value and index of value (associated emotion class) 
            _, predicted = torch.max(outputs, 1)

            # UPDATE IMAGE STATS
            total += labels.size(0)  # Num of images in current batch 
            correct += (predicted == labels).sum().item() # Compare predictions against correct label  -> sum counts nunber of correct lables (Trues)
            print(f"\r Epoch {epoch + 1}/{EPOCHS} | Batch {batch_idx}/{total_batches} ", end="")
        # CALC EPOCH STATS
        epoch_loss = running_loss / len(train_loader)  # Avg loss across all batches in current epoch  
        epoch_accuracy = 100 * correct / total 
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {epoch_loss:.4f} "
            f"Accuracy: {epoch_accuracy:.2f}% "
        )
        # --- Validation pass (used to pick the best checkpoint) ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_accuracy = 100 * val_correct / val_total
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        # --- Save best current model --- 
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), EXPERIMENT_MODEL_PATH)
            print(f"New best validation accuracy: {best_val_accuracy:.2f}% — saved.")
    return best_val_accuracy

def evaluate_model(model, device):
    """
    Evaluate the trained model on the combined test dataset.
    """
    # --- Evaluation --- 
    # How model performs on images it hasnt seen after testig 
    model.eval() # Swich model to evaluation mode
    # Reset Stats 
    correct = 0
    total = 0
    # For confusion matrix
    all_labels = []
    all_predictions = []
    # For stats on per class perdictions
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    # --- Testing --- 
    # Disable gradient calculation because we are evaluating, not training  
    with torch.no_grad():
        for images, labels in test_loader:
                    # Load images/labels 
                    images = images.to(device)
                    labels = labels.to(device)

                    # Dont Calculate gradients as we are now testing 
                    # Foward Pass through CNN
                    outputs = model(images)

                    # Select class with largest logic 
                    _, predicted = torch.max(outputs, 1) 

                    # Count total images 
                    total += labels.size(0)

                    # Count correct predictions 
                    correct += (predicted == labels).sum().item()

                    # save actual and predicted labels
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())
                    # --- Per Class Accuracy --- 
                    for label, pred in zip(labels, predicted):
                        label = label.item()
                        pred = pred.item()
                        class_total[label] += 1
                        if label == pred:
                            class_correct[label] += 1
    # --- Models accuracy ---
    test_accuracy = 100 * correct / total
    print(f"{'=' * 60}")
    for i, emotion in enumerate(EMOTION_NAMES):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
        else:
            class_acc = 0.0
        print(f"{emotion:<12} {class_correct[i]:>4} / {class_total[i]:<4} ({class_acc:>6.2f}%)")
    return all_labels, all_predictions, test_accuracy

def main():
    # --- Device ----
    device = get_device()
    print(f"Using device: {device}")

    # --- Model Instantiation ---
    # OUTPUT: 7 Logits 
    model = EmotionCNN()
    model = model.to(device) # Move to device 

    # --- Loss Function ---- 
    # OUTPUT: single number representing how wrong models prediciton are (low -> better, high -> lower)
    # Criterion is fixed formula for calculating error, what does change is weights and biases, each batch calculates new loss using this same criteria 
    criterion = nn.CrossEntropyLoss() # Compare Logits of CNN with correct class labels, CrossEntropyLoss() handles normalization

    # --- Optimizer ---- 
    # Updating these parameters each batch based off the measured loss and backprop to compute gradients is how predicitons improve 
    optimizer = optim.Adam(
        model.parameters(), # (biases, weights) 
        lr=LEARNING_RATE # By how much to update those parameters
    )

    # --- Train Model --- 
    best_val_accuracy = train_model( model, criterion, optimizer, device )

    # --- Load the BEST checkpoint (by validation accuracy) before final evaluation ---
    # The loop above saved it to disk each time val accuracy improved 
    # We need to actually load it back into `model` now.
    print(f"\nLoading best model (validation accuracy: {best_val_accuracy:.2f}%) for final evaluation...")
    model.load_state_dict(torch.load(EXPERIMENT_MODEL_PATH, map_location=device))

    # --- Evaluation ---
    all_labels, all_predictions, test_accuracy = evaluate_model( model, device ) 

    # --- Final Results ---
    print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")

    # --- Confusion Matrix ---
    plot_confusion_matrix(all_labels, all_predictions,"Combined Model — Test Confusion Matrix")

    print( f"\nExperiment model saved to:\n" f"{EXPERIMENT_MODEL_PATH}" )

# Driver
if __name__ == "__main__":
    main()