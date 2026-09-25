""" 
Handles real-time camera input for the computer vision pipeline. 
Captures webcam frames, detects faces using YOLO, and classifies each detected face using the trained EmotionCNN model.
"""

import numpy as np
import cv2 as cv #import the opencv library 
import torch
from ultralytics import YOLO
from emotion_detection.emotion_cnn import EmotionCNN
# from emotion_detection.data.dataset_loader import train_dataset
from torchvision import transforms
from collections import deque, Counter

# 1. LOAD MODELS ------
# --- Facial Detection Model --- 
facedetect_model = YOLO("/Users/laurenpalega/Documents/virtualfriend/ml/face_detection/runs/detect/train-2/weights/best.pt")  # load custom model

# --- Emotion Classification Model ---
# Create Instance of CNN architecture
emotion_model = EmotionCNN() 
# Load the parameters learned during training (weights and baises)
emotion_model.load_state_dict(  
    torch.load( "/Users/laurenpalega/Documents/virtualfriend/ml/emotion_detection/emotion_cnn_raf_affectnet_best.pth", map_location="cpu")
)
# Put model in evaluation mode 
emotion_model.eval() 

# 2. MAPPING EMOTION LABELS ------
# Convert the CNN's predicted class index into associated emotion.
# raf_emotions = { "1": "Surprise", "2": "Fear", "3": "Disgust", "4": "Happiness", "5": "Sadness", "6": "Anger", "7": "Neutral" }
# #  Create Dictionary (key: an index from 0-6, value: corresponding emotion name) 
# idx_to_emotion = {  
#     index: raf_emotions[class_name] # Update value to be corresponding emotions string representation 
#     for class_name, index in train_dataset.class_to_idx.items() # train_dataset.class_to_idx is a dict (keys: 1-7, value: 0-6) 
# }
idx_to_emotion = { 0: "Surprise", 1: "Fear", 2: "Disgust", 3: "Happiness", 4: "Sadness", 5: "Anger", 6: "Neutral" }

# 3. EMOTION PREPROCESSING ------
emotion_transform = transforms.Compose([ # Transform webcam frame to tensor shape to meet CNN architecure 
    transforms.ToPILImage(), # Converts the OpenCV/NumPy image into a PIL image.
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

emotion_history = deque(maxlen=5)

def predict_emotion(face):
    """
    Preprocess one detected face and use the trained CNN to predict its emotion.
    """
    # Convert BGR (OpenCV) -> RGB to match training data (PIL/ImageFolder)
    face_rgb = cv.cvtColor(face, cv.COLOR_BGR2RGB)
    # Preprocess the image of face to be the correct shape for the CNN to process
    # Returns transformed tensor
    face_tensor = emotion_transform(face_rgb)

    # Add batch dimension as CNN expects image input in this format 
    # (3, 128, 128) -> (1, 3, 128, 128)
    face_tensor = face_tensor.unsqueeze(0)

    # Make prediction hence no_grad (no need for loss, backpropagation, nor param updates) 
    with torch.no_grad():
        outputs = emotion_model(face_tensor) # 2d tensor: 1 image, 7 logits/ emotion predictions
        # Look across dimension 1 (the 7 emotion predictions) and return the largest logit and its index (0-6).
        # _ -> the largest logit, predicted -> its index
        _, predicted = torch.max(outputs, 1) 
        predicted = predicted.item()  # Convert tensor containing the index into a Python integer
    emotion = idx_to_emotion[predicted] # Get dict value, the emotion name, using index as key   
    return emotion

def process_face(frame, face_detection_result):
    """
        Crop each detected face, predict its emotion, and draw the detection result.
    """
    for box in face_detection_result.boxes: # For each detected bounding box/face  
        # Get Bounding box coordinates.
        x1, y1, x2, y2 = box.xyxy[0].tolist() # Unpack bounding box tensor -> coordinates -> 4 variables 
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2)) # Map bounding box coordinates as integers 

        # Crop the detected face from the frame.
        face = frame[y1:y2, x1:x2]  # array[rows, columns] -> take rows from y1 to y1, and colomns from x1 to x2 

        # Make sure the crop isn't empty.
        if face.size == 0: 
            continue

        # Pass face into predict emotion which runs data through CNN 
        emotion = predict_emotion(face)
        # Majority vote on recent emotions 
        emotion_history.append(emotion)
        stable_emotion = Counter(emotion_history).most_common(1)[0][0] 

        # Draw face bounding box on frame
        cv.rectangle(
            frame, # Image 
            (x1, y1), # Top L
            (x2, y2), # Bottom R
            (0, 255, 0), # Color
            2 # Thickness
        )
        
        # Create emotion label
        # label = f"Emotion: {emotion}"
        label = f"Emotion: {stable_emotion}"

        # Draw emotion label above face
        cv.putText(
            frame, # Image
            label, # Text 
            (x1, max(y1 - 10, 20)), # Position -> 10 px above top corner -> Max avoid negative values
            cv.FONT_HERSHEY_SIMPLEX, # Font
            0.7, # Font Scale
            (0, 255, 0),  # Color
            2 # Thickness
        )

# 4. OPEN CAMERA AND DISPLAY RESULTS ------
cap = cv.VideoCapture(0) #function to open default webcame (0) 

if not cap.isOpened(): # Webcam opened correctly? check input source worked first
    print("Cannot open cameqra")
    exit()

while True: # Loop so its not a static image, but a video feed
    # Ret is a boolean variable which returns true if the frame is available
    # Frame is numpy array which holds the image data in BGR order
    ret, frame = cap.read() # Capture frame-by-frame

    if not ret:    # If frame is read correctly ret is True
        print("Can't receive frame. Exiting ...")
        break

    frame = cv.flip(frame, 1) # Flip the frame horizontally using "1" (mirror image) 

    # --- Face Detection Processing ---
    results = facedetect_model(frame) # Pass the frame to the model for inference 
    face_detection_result = results[0] # [0] <- Resulta object for webcame frame

    # --- Emotion Detection Processing ---
    process_face(frame, face_detection_result) 

    # --- Display annoted frame --- 
    cv.imshow('annotated_frame', frame)

    # --- Exit Loop Logic --- 
    key = cv.waitKey(1) & 0xFF 
    if key == ord("q") or key == 27:    # Press Q or ESC to quit.
        break

cap.release() # Release webcam 
cv.destroyAllWindows() # Close the window displaying the video feed