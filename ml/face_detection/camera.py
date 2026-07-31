import numpy as np
import cv2 as cv #import the opencv library 
from ultralytics import YOLO

model = YOLO("/Users/laurenpalega/Documents/virtualfriend/ml/face_detection/runs/detect/train-2/weights/best.pt")  # load custom model

cap = cv.VideoCapture(0) #function to open default webcame (0)  
if not cap.isOpened(): #webcame opened correctly? check input source worked first. 
    print("Cannot open camera")
    exit()
while True: #loop so its not a static image, but a video feed.
    #ret is a boolean variable which returns true if the frame is available. 
    #frame is numpy array which holds the image data in BGR order. docs/ml.txt for more info. 
    ret, frame = cap.read() # Capture frame-by-frame

    if not ret:    # if frame is read correctly ret is True
        print("Can't receive frame. Exiting ...")
        break

    frame = cv.flip(frame, 1) #flip the frame horizontally using "1" (mirror image) 

    results = model(frame) #pass the frame to the model for inference 
    annotated_frame = results[0].plot() #draw the bounding boxes on the frame 

    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    # transform frame to gray scale 
    # cv.imshow('frame', frame)    # Display the resulting frame
    cv.imshow('annotated_frame', annotated_frame)    # Display the resulting frame with bounding boxes 

    if cv.waitKey(1) == ord('q'): #ord coverts to unicode value
        #waitKey(x) beifly checks for input every x milliseconds, returns the unicode value of (if!) the key pressed 
        break

cap.release() #release webcam 
cv.destroyAllWindows() #close the window displaying the video feed