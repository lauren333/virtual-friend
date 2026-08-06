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


## 3. Emotion Recognition  
### 3.1 Images as Data  
- Essentially an image is a matrix of numbers (pixel values)
- Grayscale image:
    - Height × Width
- RGB Image: 
    - They have depth -> RGB 
    - Height × Width x Channels 
    - Channels = 3 for RGB (Red, Green, Blue) 

### 3.2 Convolutional Neural Networks 
- CNNs are neural networks designed for visual data 
- They learn patterns directly from raw pixels by applying convolution operations

- Consisting of two main parts:
    1. Feature Extractor
        - Convolution layers
        - Activation functions (ReLU)
        - Pooling layers
        - Trades: Spacial resolution for richer feature representation 
    2. Classifier
        - Flatten layer
        - Fully connected (Linear) layers
        - Maps the extracted features to the final emotion classes.

### 3.3 Convolution Operation
[Source](https://www.youtube.com/watch?v=YGILT182T6w)
#### Main Idea: 
- Images contain spacial structure and local context -> which pixels are near eachother is important 
- Slide kernel across the image -> create feature maps 
#### Kernel 
- One kernel detects one kind of pattern, for multiple features -> mutliple kernels 
- Each slide independently, each yeild there own 2d output -> feature map
- Output size: H' = H - k + 1  

#### Multi-channel input: 
- One layer of pixel values: gray scale image H * w 
- Real images have depth RGB -> H*W*C (c is number of channels)
- Now kernels need to match that depth -> k*k*c volume and expands through all channels 
- Kernel still slides through spacial dimensions, but at each position processes all channels at once producing 2d output H' * W' 
- And when using multiple kernels k' of them -> stack all the ouputs to get H' * W' * C'  

#### Multiple CNN Layers
- In CNN we dont dont this only once, we do it numerous times, the ouput of one layer becomes input of another -> Pooling layers progressively reduce the spatial dimensions while convolution layers learn increasingly complex features.  
- At the end we take final compacted volume and flatten it into a vector -> a learned feature vector a representation that the network has learned to extract from raw pixels
#### Pooling: 
- After each convultional layer we apply pooling to shrink spacial dimension 
- number of channels remains unchanged 
- Network tradeoff: spacial resolution for richer feature representations 
#### Why CNN work: 
- The assumptions that make the CNNs work for visual data(inductive biases): 
- Local Connectivity: 
    - Each output neuron only looks at small local patch of input 
    - region covered by kernel 
- Translation Equivarience: 
    - If you shift input by x pixels the output shifts x pixels 
    - Perserves spacial realtionships!  
- Parameter Sharing: 
    - Learn one set of weights that works everywhere 
    - reduces parameters and makes network easier to train 
- Translation invariance: 
    - Position doesnt matter for final prediciton  
- Hierarchical features: 
    - Each layer builds from previous one -> yielding powerful representation

### 3.4 How Nueral Networks Learn - Gradient Descend
[Source](https://www.youtube.com/watch?v=IHZwWFHWa-w) 
#### Forward Pass
- Each pixelvalue (on the grid/image) becomes an in the input layer of the network
- Each activation of the nuerons in the following hiden layers are calculated by the weighted sum of all the activations in previous layer plus the bias. 
    - Compose that sum for example by the sigmoid squish-ification 
- weight and biases control what the networka actually does -> how it learns is by tweaking these values 
#### Learning Process
- Want: and algorithm 
    - show a bunch of training data (with labels)
    - it adjusts its weights and biases enough to improve its performance on training data
    - -> goal: generalizes images beyond that training data 
#### Weights and Biases
- Essentially calculus: finding minimia of certain function 
    - Each nueron is connected to all the nueron in previous layers 
    - weights define its activation are like the strengths of those connections -> larger weights mean that input has more influence on nuerons activation 
    - Bias is just another learnable parameter (we perhaps can think of it as indiction of wether that nueon is active or inactive) 
    - start with random wights and baises 
#### Cost Function
- -> define cost function 
    - what is the cost of the difference between "bad" result and expected -> one way is Mean Sqaured Error, add up squares of differences between bad output and what you want it to be
    - small when good, large when bad 
    - then we find the average cost over all the tens and thousands of training examples -> defines how well the network classifies
    - describes how good or bad those biases/weights are essentially  
    - we need to tell it how to change from this to better the algorithm -> we want to minimize the cost function
#### Gradient Descent
- where do we step from the current weight to minimize error and improve the cost function results (by miniziming cost -> better performance on all samples)
    - Algorithm for computing this gradient efficiently -> Backpropagation 
    - Gradient descent: finding valley in graph 
    - Gradient Vector of cost function 
        - encodes relative importance of each weight and bais 
        - which changes to wich weights have more impact persay 
#### Backpropagation
- Algorithm for computing this gradient efficiently -> Backpropagation 

#### Weight Update
- how weights actually change: new_weight = old_weight - learning_rate × gradient

### Dataset 
- [source](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset/data):