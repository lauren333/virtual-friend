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
from emotion_cnn import EmotionCNN
from data.dataset_loader import train_loader, test_loader

# --- Training Configuration ---- 
NUM_CLASSES = 7 # One logit / each emotion class 
LEARNING_RATE = 0.001 # How large the optimizer's parameter updates -> how much to update weights by -> weight' = weight - learning_rate * gradient 
EPOCHS = 1 # Num of complete passes through training set

# --- Device ---- 
# Apple M4 Macs can use Apple's Metal Performance Shaders (MPS) -> perform support ops on GPU instead of CPU 
if torch.backends.mps.is_available():
    device = torch.device("mps")
else: 
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- Model Instantiation ---- 
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

# --- Training Loop ---- 
# So each batch of images goes through 4 processes 
# -> pass in images, compute loss, backpropgate to get gradients, then update parameters (+ update stats) 
for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0 # Accumulate loss across epochs to calculate avg loss
    correct = 0 # Num of correctly classified images 
    total = 0 # Num of classifed images 

    for images, labels in train_loader:
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

    # CALC EPOCH STATS
    epoch_loss = running_loss / len(train_loader)  # Avg loss across all batches in current epoch  
    epoch_accuracy = 100 * correct / total 
    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {epoch_accuracy:.2f}%"
    )

# --- Evaluation --- 
# How model performs on images it hasnt seen after testig 

model.eval() # Swich model to evaluation mode

# Reset Stats 
correct = 0
total = 0

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
# Models accuracy 
test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# --- Save Trained Model --- 
torch.save(
    model.state_dict(),
    "emotion_cnn.pth"
)
print("Model saved to emotion_cnn.pth") 