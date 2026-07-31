# Machine Learning / Computer Vision Notes

## 1. Capturing Video from Camera
- Tools used: OpenCV and its documentation https://docs.opencv.org/4.13.0/dd/d43/tutorial_py_video_display.html 
- Frame is a numpy array. 
- An image is represented as a large array of numbers 
- OpenCV normally stores colors as blue green red (BGR)
- Each pixel stores 3 numbers: B,G,R which together make that pixels color value 
- So each cell has this information -> each cell represents a pixel, you make an image "collecting" these in rows and colomns
- The number of rows and colomns corresponds to the height and width of an image pixel-wise

## 2. Face Detection 
### Goal
- Detect an object, in my case, faces using bounding boxes

### Initial Approach - OpenCv Haar Cascade 
- My original plan was to train an OpenCV Haar Cascade classifier using my own dataset. I collected and organised images and intended to manually create the bounding boxes using OpenCV's annotation and training tools.
- The utilities required to train Haar Cascade classifiers (opencv_createsamples and opencv_traincascade) were removed from OpenCV 4.x and are only available in the legacy OpenCV 3.4 branch. Although the trained cascade models remain compatible with newer OpenCV versions, the training itself must be performed using the older tools.
- I attempted to set up an OpenCV 3.4 environment specifically for training, with the intention of using the resulting model in my newer OpenCV installation. However, because my development environment uses a recent version of Xcode and modern system libraries, I encountered compatibility and build issues with the legacy OpenCV tools. 
- Rather than continuing to troubleshoot outdated software, I evaluated other options. I selected YOLO because it is actively maintained and provides higher accuracy than traditional Haar Cascade classifiers for face detection. 

### Approach - YOLO 
- You only look once: Convolutional neural network designed for object detection.
- Nueral network learns visual features from labelled training data -> learns location of objects (bounding boxs) and class. 
- Advantages: 
    - Automatically learns features from data rather then relying on manually engineered features.
    - Robust: to changes in lightening, oritnetation, scale, occulsion. 
    - Higher detection accuracy than Haar Cascade classifiers. 
    - Maintained and well documented. 
- The pretrained YOLO11 Nano (yolo11n.pt) model was fine tuned using transfer learning allowing the model to reuse features learned from a large object detection dataset while adapting specifically to face detection.
- Tools used: YOLO Documentation for model training and usage.  

### Dataset 
#### Initial Dataset 
- The initial model was trained using the Face Detection dataset from Roboflow Universe. 
    - Source: Face Detection Dataset ([Roboflow Universe](https://universe.roboflow.com/yolo-training-hwj0p/face-detection-3n1j3))
    - Dataset size: 252 labelled images. 
- All images were extracted from videos of people sitting on a bus. 
    - -> many nearly identical frames, faces that were small and far from the camera, limited variation in pose, lighting, background, and aswell there was a lack of diversity of individuals. 
- The trained model did not accurately detect my face in the live webcam stream and frequently missed faces in the test images. This was likely due to both the limited size of the training dataset (252 images) and its lack of diversity, making it difficult for the model to generalise to new faces and environments.  

#### Final Dataset 
- The model was then retrained on a larger more diverse dataset. 
- Dataset: Face Detection Computer Vision Dataset ([Roboflow Universe](https://universe.roboflow.com/shashank-ghodke-z8dxx/face-detection-mik1i-gqsox?)) 
- I used 400 images from the 1k+ dataset and then changed the split to 8:1:1 for train, valid, test. 
- Compared with the initial dataset, it contained much greater variation in face sizes, individuals, viewing angles, lighting conditions, and backgrounds. This improved the model's ability to generalise and resulted in substantially better detection performance on both the test images and the live webcam feed.

### Model Training 
The model was trained using the following command:

```bash
yolo detect train \
model=yolo11n.pt \              # Pretrained YOLO11 Nano model
data=datasets/faces/data.yaml \ # Dataset configuration file
epochs=100 \                    # Number of complete training passes
imgsz=512 \                     # Resize all images to 512 × 512 pixels
batch=20 \                      # Images processed before each weight update
name=face_detector_v1           # Name of the training run
```

### Model Testing
**Single image**
```bash
yolo detect predict model=runs/detect/train-2/weights/best.pt source=test.jpg
```
**Live webcam**
```bash
yolo detect predict model=runs/detect/train-2/weights/best.pt source=0
```

### Results
- Evaluation metrics: 
    - **Precision: 0.993**
        - When the model said it had found a face, it was correct **99.3% of the time**, meaning it rarely detected something that wasn't actually a face.
    - **Recall: 0.950**
        - Model successfully found **95.0% of all the faces** in the test images, indicating it only missed a small number of faces.
    - **mAP50: 0.989**
        - Overall measure of how accurately the model detects and places bounding boxes around faces. A score of **0.989** indicates good overall detection performance.  

### Conlusion
- The detector performed reliably on the live webcam feed but was less accurate on some static images containing faces at extreme angles or with partial occlusion.
- These limitations were observed during testing on additional images outside the training and test datasets.
- Using the full 1,000+ image dataset would likely improve the model's ability to generalise. However, due to time and computational constraints, a subset of 400 images was used.
- Despite these limitations, the detector performed accurately enough for the final application.