# Emotion Recognition: Machine Learning Documentation

**Author:** Lauren Nicole Palega

**Domain:** Computer Vision · Deep Learning

**Stack:** Python · PyTorch · OpenCV · Ultralytics YOLO11

---
## Table of Contents
1. [Project Context and Overview](#1-project-context-and-overview)
2. [Computer Vision Background](#2-computer-vision-background)
3. [Face Detection (YOLO11n)](#3-face-detection-yolo11n)
4. [Model 1: Static-Image CNN](#4-model-1-static-image-cnn)
5. [Pivot: From Static Images to Video](#5-pivot-from-static-images-to-video)
6. [Model 2: CNN + GRU](#6-model-2-cnn--gru)
7. [Return to Model 1: Expanded Dataset](#7-return-to-model-1-expanded-dataset)
---

# 1. Project Context and Overview
## 1.1 Context
This document covers the **machine learning component** of a larger project: a small AI companion that reacts to the user's emotions. The ML system is the part that "sees" the user, detecting a face and recognising which emotion it is showing. This write-up documents that component only.

## 1.2 Project Goal
Develop a computer vision system capable of recognising human facial emotions.

The project began with a static-image convolutional neural network (CNN), which classified individual facial images into seven emotion classes. The project was then extended to video by combining a CNN with a gated recurrent unit (GRU), allowing the model to process sequences of facial frames and learn temporal information.

Following the video experiments, I returned to the original image-based approach and expanded the training data to investigate whether improved data quality and diversity could produce better generalisation.

The project therefore explores two related approaches:
1. **Static-image emotion recognition** using a CNN.
2. **Video-based emotion recognition** using a CNN + GRU.

The development process involved experimenting with model architecture, data augmentation, dropout, class imbalance, sequence length, and dataset composition.
## 1.3 System pipeline
```
Webcam / video → Face detection (YOLO11n) → Face crop → Emotion model → Emotion prediction
```

# 2. Computer Vision Background
## 2.1 Capturing Video with OpenCV
- Tools used: OpenCV and its documentation [[source](https://docs.opencv.org/4.13.0/dd/d43/tutorial_py_video_display.html)]
- A video is comprised of individual images, or frames.
- When using OpenCV to capture video, a frame is a NumPy array. 
- An image can then be understood as a large array of numbers (numerical pixel values). 
- OpenCV normally represents color images using BGR color order (Blue, Green, Red).
- Each pixel stores 3 numbers: B,G,R which together make that pixels color value.
- So each cell has this information -> each cell represents a pixel, you make an image "collecting" these in rows and colomns.
- The number of rows and colomns corresponds to the height and width of an image pixel-wise.

## 2.2 Images as data
- Essentially an image is a matrix of numbers (pixel values).
- **Grayscale image:** `Height × Width`.
- **RGB image:** has depth, `Height × Width × Channels`, where Channels = 3 (Red, Green, Blue).

## 2.3 Convolutional neural networks
- CNNs are neural networks designed for visual data. 
- They learn patterns directly from raw pixels by applying convolution operations, and consist of two main parts:
1. **Feature extractor**
    - Convolution layers
    - Activation functions (ReLU)
    - Pooling layers
    - Trades spatial resolution for richer feature representations
2. **Classifier**
    - Flatten layer
    - Fully connected (linear) layers
    - Maps the extracted features to the final emotion classes

## 2.4 The convolution operation
*Reference: [video tutorial by DataMListic](https://www.youtube.com/watch?v=YGILT182T6w)*

**Main idea.** Images contain spatial structure and local context, so which pixels are near each other matters. A kernel slides across the image and creates feature maps.

**Kernels.**
- One kernel detects one kind of pattern; to detect multiple features use mutliple kernels. 
- Each slide independently and each yeild there own 2d output -> a feature map. 
- Output size: `H' = H - k + 1`.

**Multi-channel input.**
- A grayscale image is a single layer of pixel values (`H × W`). Colored images have depth (`H × W × C`, where C is the number of channels), which makes them 3D tensors.
- A **Tensor** is a multidimensional array of numerical values; a 3D tensor can be visualized as a stack of matrices. 
- Now kernels need to match that depth, so they are `k × k × C` volumes that expands through all channels. 
- The kernel still slides across the spacial dimensions, but at each position it processes all channels at once producing 2D output of size `H' × W'`.
- When using multiple kernels (`K'` kernels), the outputs are stacked to give an `H' × W' × K'` volume.

**Multiple CNN layers.**
- The convolution is not applied just once. It is repeated many times, with the output of one layer becoming the input of the next.
- Pooling layers progressively reduce the spatial dimensions while convolution layers learn increasingly complex features.  
- At the end, the final compacted volume is flattened into a vector -> a learned feature vector, a representation that the network has learned to extract raw pixels.

**Pooling.**
- After each convultional layer we apply pooling to shrink spacial dimension. 
- Number of channels remains unchanged. 
- Network tradeoff: the network trades spacial resolution for richer feature representations. 

**Why CNNs work (inductive biases).**
- The assumptions that make the CNNs work for visual data.  
| Property | Meaning |
| -------- | ------- |
| Local connectivity | Each output neuron only looks at a small local patch of the input (the region covered by the kernel). |
| Translation equivariance | If the input shifts by x pixels, the output shifts by x pixels. Spatial relationships are preserved. |
| Parameter sharing | One set of weights is learned and works everywhere, which reduces the number of parameters and makes the network easier to train. |
| Translation invariance | Position does not matter for the final prediction. |
| Hierarchical features | Each layer builds on the previous one, yielding a powerful representation. |

## 2.5 How neural networks learn: gradient descent
*Reference: [video tutorial by 3Blue1Brown](https://www.youtube.com/watch?v=IHZwWFHWa-w)*
#### Forward Pass
- Each pixel value (on the image) becomes an in the input activation of the network. 
- Each activation of the nuerons in the following hiden layers are calculated by the weighted sum of all the activations in previous layer plus the bias, passed through an activation function (Compose that sum for example by the sigmoid squish-ification).
- Weight and biases control what the network actually does, learning means tweaking these values. 
- In my CNN, images are passed through the network in batches rather than one image at a time. With `batch_size = 32`, the input is a 4D tensor of shape `(32, 3, 128, 128)`.
    - The 32 images are processed together using the same weights, but each image is processed independently. 
- The forward pass produces the model's raw outputs, called **logits**. In my model there are 7 logits per image, one per emotion class.

### Learning process
The goal is an algorithm that is shown labelled training data, adjusts its weights and biases to improve its performance on that data, and, most importantly, **generalises** to images beyond the training data.

Training repeats these steps for each batch:
1. **Forward pass:** send in images and labels, producing predictions (logits).
2. **Calculate loss:** measure how wrong the predictions are.
3. **Backpropagation:** calculate the gradient showing how the weights and biases should change to reduce the loss (which direction to move to reduce loss).
4. **Optimiser step:** update the weights and biases using the gradient (direction) and learning rate (step size). 
5. **Update statistics:** track loss and accuracy.

This repeats for every batch in the training set. One full pass through the training set is one **epoch**, and the model trains for multiple epochs so it can incrementally improve.

### Weights and biases
- This is essentially calculus: finding the minimum of a function. 
- Each nueron is connected to all the nueron in previous layers. Weights are like the strengths of those connections, so larger weights mean that input has more influence on nuerons activation. The bias is another learnable parameter.
- Training starts with random weights and baises. 
- The weights and biases are the parameters that the optimizer changes during training. 
- The loss function itself is not updated. It is a fixed formula that measures how wrong the current predictions are. It is recalculated each batch using the same criterion but with the newly updated weights and biases, which is how the model improves.

### Cost function
- A cost function measures the difference between a "bad" result and the expected one. One option is Mean Squared Error: the sum of squared differences between the actual output and the desired output. It is small when the network is good and large when it is bad.
- Averaging the cost over the tens of thousands of training examples defines how well the network classifies, and describes how good or bad the current weights and biases are. Aka how well the network classifies. 
- To improve, the network must be told how to change, and the aim is to **minimise the cost function**.
- In my implementation I use **CrossEntropyLoss** instead of Mean Squared Error. It compares the model's logits with the correct class labels and produces a single loss value. Lower loss generally means the predictions are improving.

### Gradient descent
- The question is which way to step from the current weights to reduce the cost. Gradient descent is like finding the valley of a graph.
- The gradient vector of the cost function encodes the relative importance of each weight and bias, in other words which changes to which weights have more impact.
- Moreover, gradient tells us which direction each weight and bias should move to reduce the loss. 
- The **learning rate** controls how large a step is taken; a smaller learning rate makes smaller updates.

### Backpropagation
- The algorithm that computes the gradient efficiently.
- After the loss is calculated, backpropagation works backwards through the network to calculate the gradient of the loss with respect to each trainable weight and bias. 
- The optimiser then uses these gradients and the learning rate to update the weights and biases.

### Weight update
- How weights change: `new_weight = old_weight - learning_rate × gradient`.
- In my implementation the **Adam** optimiser performs the updates (`optimizer.step()`).

**Overall idea**
| Concept | Role |
| ------- | ---- |
| Loss | How wrong the current predictions are |
| Gradient | Which direction reduces the loss |
| Learning rate | How large the step is |
| Optimiser | Applies the update to the weights and biases |

Repeating these updates across batches and epochs lets the model gradually learn better visual patterns and improve its predictions.


# 3. Face Detection (YOLO11n)
## 3.1 Goal
- Detect an object, in my case, faces using bounding boxes so only that region is passed into the emotion model. 

## 3.2 Initial approach: OpenCV Haar Cascade
- My original plan was to train an OpenCV Haar Cascade classifier using my own dataset. I collected and organised images and intended to manually create the bounding boxes using OpenCV's annotation and training tools.
- The utilities required to train Haar Cascade classifiers (`opencv_createsamples` and `opencv_traincascade`) were removed from OpenCV 4.x and are only available in the legacy OpenCV 3.4 branch. Although the trained cascade models remain compatible with newer OpenCV versions, the training itself must be performed using the older tools.
- I attempted to set up an OpenCV 3.4 environment specifically for training, with the intention of using the resulting model in my newer OpenCV installation. However, because my development environment uses a recent version of Xcode and modern system libraries, I encountered compatibility and build issues with the legacy OpenCV tools. 
- Rather than continuing to troubleshoot outdated software, I evaluated other options. I selected **YOLO** because it is actively maintained and provides higher accuracy than traditional Haar Cascade classifiers for face detection. 

## 3.3 Selected approach: YOLO
- **You Only Look Once:** a Convolutional neural network designed for object detection.
- Nueral network learns visual features from labelled training data -> learns location of objects (bounding boxs) and class. 
- Advantages: 
    - Automatically learns features from data rather then relying on manually engineered features.
    - Robust: to changes in lightening, oritnetation, scale, occulsion. 
    - Higher detection accuracy than Haar Cascade classifiers. 
    - Maintained and well documented. 
- The pretrained YOLO11 Nano (`yolo11n.pt`) model was fine tuned using transfer learning allowing the model to reuse features learned from a large object detection dataset while adapting specifically to face detection.
- Tools used: YOLO Documentation for model training and usage.  

## 3.4 Dataset
### Initial dataset
- The initial model was trained using the Face Detection dataset from Roboflow Universe. 
    - Source: Face Detection Dataset ([Roboflow Universe](https://universe.roboflow.com/yolo-training-hwj0p/face-detection-3n1j3))
    - Size: **252 labelled images**.
- All images were extracted from videos of people sitting on a bus. 
    - -> many nearly identical frames, faces that were small and far from the camera, limited variation in pose, lighting, background, and aswell there was a lack of diversity of individuals. 
- The trained model did not accurately detect my face in the live webcam stream and frequently missed faces in the test images. This was likely due to both the limited size of the training dataset (252 images) and its lack of diversity, making it difficult for the model to generalise to new faces and environments.  

### Final dataset
- The model was then retrained on a larger more diverse dataset. 
- Dataset: Face Detection Computer Vision Dataset ([Roboflow Universe](https://universe.roboflow.com/shashank-ghodke-z8dxx/face-detection-mik1i-gqsox?)) 
- I used **400 images** images from the 1k+ dataset and then changed the split to 8:1:1 for train, valid, test. 
- Compared with the initial dataset, it contained much greater variation in face sizes, individuals, viewing angles, lighting conditions, and backgrounds. This improved the model's ability to generalise and resulted in substantially better detection performance on both the test images and the live webcam feed.

## 3.5 Training and testing
```bash
yolo detect train \
model=yolo11n.pt \              # Pretrained YOLO11 Nano model
data=datasets/faces/data.yaml \ # Dataset configuration file
epochs=100 \                    # Number of complete training passes
imgsz=512 \                     # Resize all images to 512 × 512 pixels
batch=20 \                      # Images processed before each weight update
name=face_detector_v1           # Name of the training run
```

```bash
# Single image
yolo detect predict model=runs/detect/train-2/weights/best.pt source=test.jpg
```
```bash
# Live webcam
yolo detect predict model=runs/detect/train-2/weights/best.pt source=0
```

## 3.6 Results
| Metric | Value | Interpretation |
| ------ | ----: | -------------- |
| Precision | **0.993** | When the model said it found a face, it was correct 99.3% of the time, so it rarely detected something that was not a face. |
| Recall | **0.950** | Model successfully found **95.0% of all the faces** in the test images, indicating it only missed a small number of faces. |
| mAP50 | **0.989** | Overall measure of how accurately the model detects and places bounding boxes around faces; 0.989 indicates good overall detection. |

## 3.7 Conclusion
- The detector performed reliably on the live webcam feed but was less accurate on some static images containing faces at extreme angles or with partial occlusion.
- These limitations were observed during testing on additional images outside the training and test datasets.
- Using the full 1,000+ image dataset would likely improve the model's ability to generalise. However, due to time and computational constraints, a subset of 400 images was used.
- Despite these limitations, the detector performed accurately enough for the final application.

# 4. Model 1: Static-Image CNN
## 4.1 Dataset
[RAF-DB](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset/data): seven emotion classes.

| Class | Train | Test |
| ----- | ----: | ---: |
| 1 Surprise | 1,290 | 329 |
| 2 Fear | 281 | 74 |
| 3 Disgust | 717 | 160 |
| 4 Happiness | 4,772 | 1,185 |
| 5 Sadness | 1,982 | 478 |
| 6 Anger | 705 | 162 |
| 7 Neutral | 2,524 | 680 |
| **Total** | **12,271** | **3,068** |

RAF-DB is heavily imbalanced. There are few images of Fear, Anger, and Disgust, and Happiness has approximately **17 times** more training examples than Fear.

## 4.2 Experiment 1: Baseline CNN
**Objective:** evaluate the initial CNN architecture without augmentation or regularisation.
| Epochs | Test Accuracy |
| ------ | ------------: |
| 1      |        63.46% |
| 10     |        71.61% |
| 20     |        69.88% |
| 25     |        70.93% |
 
Final training result: Epoch 25/25, loss 0.0438, train accuracy 98.73%, test accuracy 70.93%.

**Finding:** the model learned the training set extremely well but generalised poorly to unseen images. The large gap between training and test accuracy indicates substantial overfitting.

## 4.3 Experiment 2: Improving generalisation
**Objective:** reduce overfitting through data augmentation and dropout regularisation.
 
| Experiment        | Train Accuracy | Test Accuracy |
| ----------------- | -------------: | ------------: |
| Original CNN      |         98.73% |        70.93% |
| + Augmentation    |         91.95% |        74.71% |
| + Dropout 0.5     |         74.95% |        74.38% |
| **+ Dropout 0.3** |     **83.38%** |  **75.46%** |

**Findings**
- Data augmentation significantly improved generalization.
- Dropout reduced overfitting by preventing the network from relying on specific neurons.
- A dropout rate of 0.3 produced the best balance between training and testing performance.
- Test accuracy improved by approximately 4.5 percentage points compared with the original model.

## 4.4 Experiment 3: Class-level performance analysis
| True class | Correct | Total | Approx. recall |
| ---------- | ------: | ----: | -------------: |
| 1 Surprise | 270 | 329 | 82.1% |
| 2 Fear | 31 | 74 | 41.9% |
| 3 Disgust | 49 | 162 | 30.2% |
| 4 Happiness | 1,077 | 1,185 | 90.9% |
| 5 Sadness | 305 | 478 | 63.8% |
| 6 Anger | 113 | 162 | 69.8% |
| 7 Neutral | 512 | 680 | 75.3% |

**Findings**
- There is a large imbalance between the number of examples in each class.
- The model does very well on Happiness (90.9%) but poorly on Fear (41.9%) and Disgust (about 30%).
- These emotions are frequently confused with other facial expressions, and those classes also have substantially fewer examples.

**Is it my model or the dataset?**
- Fear has few images and performs poorly.
- Disgust has a moderate number of images and still performs poorly.
- Anger has a similar number to Disgust and performs much better.

## 4.5 Experiment 4: Dataset imbalance investigation
The class distribution in Section 4.1 shows that RAF-DB is heavily imbalanced, with far fewer Fear, Anger, and Disgust images than Happiness. This imbalance likely contributes to the lower recall for Fear and Disgust.

## 4.6 Experiment 5: Class weighting
**Objective:** improve recognition of underrepresented emotions by assigning larger loss penalties to minority classes.
| Experiment         | Test Accuracy |
| ------------------ | ------------: |
| Dropout 0.3        |        75.46% |
| Soft Class Weights |        69.82% |

**Finding:** class weighting improved some minority classes but substantially reduced overall accuracy, so it was not selected for the final model.

## 4.7 Experiment 6: Deeper CNN architecture
**Objective:** increase feature-extraction capacity by adding a fourth convolutional layer.

| Model | Train accuracy | Test accuracy |
| ----- | -------------: | ------------: |
| Previous best (augmentation + dropout 0.3) | 77.26% | 75.62% |
| **Deeper CNN** | **84.53%** | **77.61%** |
 
*The "previous best" row is a re-run of the Experiment 2 configuration, so its numbers differ slightly from that table.*
| Emotion | Previous recall | Deeper CNN recall | Change |
| ------- | --------------: | ----------------: | -----: |
| Surprise | 82.1% | 79.6% | -2.5% |
| Fear | 41.9% | 51.4% | **+9.5%** |
| Disgust | 30.2% | 30.6% | +0.4% |
| Happiness | 90.9% | 90.7% | -0.2% |
| Sadness | 63.8% | 72.0% | **+8.2%** |
| Anger | 69.8% | 64.2% | -5.6% |
| Neutral | 75.3% | 74.9% | -0.4% |

**Findings**
- Adding a fourth convolutional layer increased overall test accuracy from 75.62% to **77.61%**.
- The deeper architecture improved feature extraction and allowed the network to learn more complex facial patterns.
- Fear improved substantially (41.9% to 51.4% recall), and Sadness also improved significantly (63.8% to 72.0%).
- Disgust remained the most difficult emotion, showing almost no improvement despite the architectural changes.
- Small decreases were observed for Surprise, Anger, and Neutral.
- Happiness remained consistently strong at approximately 91% recall.

## 4.8 Reproducibility check
Re-running the final configuration with the current code (20 epochs) gave train accuracy **77.24%** and test accuracy **77.18%**, consistent with the 77.61% result. The model is saved as `emotion_cnn_baseline.pth`.

## 4.9 Interim conclusion
The best static model (4 convolutional layers, augmentation, dropout 0.3) reaches about **77.6%** test accuracy. Happiness, Surprise, and Neutral are recognised well, while Fear and Disgust remain weak. It also has a practical limitation: in real-time video, each frame is classified independently, so predictions are unstable. This led to the pivot to video.

---

# 5. Pivot: From Static Images to Video
The first model treated each image independently. Although the static-image CNN achieved reasonable performance on individual facial expressions, I wanted to investigate whether using video could provide additional information.

The motivation for the second model was to determine whether **temporal information** could improve emotion recognition. Facial expressions change over time, so a sequence of frames may contain information that is not available from a single frame.

The model therefore changed from:
| | Static model | Video model |
| - | ------------ | ----------- |
| Training task | Classify one image | Classify a short sequence of face images |
| Live demo | Predict every frame | Collect recent frames, run the sequence model, then display the prediction |
| | Image → CNN → Emotion | Video → 16 face frames → CNN → GRU → Emotion |

**What is a GRU?** A type of artificial neural network designed to process data that changes over time. It lets the model capture how expressions change, rather than looking at isolated frames. This makes the model not only spatial but **spatio-temporal**: the CNN learns what each frame looks like, and the GRU learns how those features change over the sequence.

---

# 6. Model 2: CNN + GRU
## 6.1 Dataset
[RAVDESS](https://www.kaggle.com/datasets/orvile/ravdess-dataset) video data, split **by actor**, so the model is tested on people it has never seen.

| Variable | Value |
| -------- | ----: |
| Total usable videos | 2,076 |
| Training / validation / test | 1,408 / 316 / 352 |
| Actor split (train / val / test) | 1–16 / 17–20 / 21–24 |
| Classes | 7 |
| Frames per video | 16 |
| Image size | 128 × 128 |

| Class | Train | Validation | Test |
| ----- | ----: | ---------: | ---: |
| Surprise | 128 | 32 | 32 |
| Fear | 256 | 56 | 64 |
| Disgust | 128 | 32 | 32 |
| Happiness | 256 | 56 | 64 |
| Sadness | 256 | 56 | 64 |
| Anger | 256 | 56 | 64 |
| Neutral | 128 | 28 | 32 |

The classes are unbalanced: Surprise, Disgust, and Neutral have half as many videos as the others.

## 6.2 Experiment 1: Baseline CNN + GRU 
**Objective:** Establish a baseline video model by combining a CNN with a GRU. 

### How a video moves through the model
| Step | Operation | Tensor shape |
| ---- | --------- | ------------ |
| 0 | Batch of videos | `[B, 16, 3, 128, 128]` |
| 1 | Merge batch and time so the CNN sees every frame as an ordinary image | `[B × 16, 3, 128, 128]` |
| 2 | CNN feature extraction (one 256-D vector per frame) | `[B × 16, 256]` |
| 3 | Restore the sequence structure | `[B, 16, 256]` |
| 4 | GRU reads the 16 vectors in order, updating its hidden state at each frame; the final hidden state summarises the video | `[B, 128]` |
| 5 | Linear classifier | `[B, 7]` (emotion logits) |

### Architecture
| Component | Configuration |
| --------- | ------------- |
| CNN | 4 convolutional layers (3 → 32 → 64 → 128 → 256 channels), ReLU, 3 max-pool layers; the resulting `256 × 16 × 16` volume is flattened (65,536 values) into a fully connected layer of 256 units |
| GRU | 1 layer, input size 256, hidden size 128, unidirectional; the last hidden state is used as the video representation |
| Classifier | Linear, 128 → 7 |
| Loss / optimiser | CrossEntropyLoss / Adam, learning rate 0.0003 |
| CNN initialisation | From scratch |


**5-epoch run.** 
| Epoch | Train loss | Train acc. | Val. loss | Val. acc. |
| ----: | ---------: | ---------: | --------: | --------: |
| 1 | 1.9070 | 17.68% | 1.9007 | 17.72% |
| 2 | 1.7344 | 30.89% | 1.7382 | 28.48% |
| 3 | 1.3372 | 49.15% | 1.7260 | 35.13% |
| 4 | 1.0588 | 62.50% | 1.7405 | 34.49% |
| 5 | 0.8240 | 73.15% | 1.9572 | 40.51% |

**Findings** 
- The first training run showed that the model learned the training data quickly, while validation performance began to stagnate and validation loss began increasing. 
- Final test accuracy on the 352 test videos: **40.91%**.

**10-epoch run.** 
| Epoch | Train acc. | Validation acc. |
| ----: | ---------: | --------------: |
| 1 | 23.3% | 23.4% |
| 2 | 38.2% | 29.8% |
| 3 | 53.1% | 35.4% |
| 4 | 66.6% | **42.4%** |
| 5 | 78.5% | **43.0%** |
| 6 | 85.1% | 39.6% |
| 7 | 89.4% | 38.3% |
| 8 | 93.7% | 41.1% |
| 9 | 96.2% | 42.1% |
| 10 | 97.2% | 42.4% |

**Findings** 
- Training accuracy kept climbing while validation accuracy remained lower. 

## 6.3 Experiment 2: GRU and Regularisation Ablations 
**Objective:** Investigate whether changes to the temporal model or regularisation could improve generalisation.

| Experiment | Best validation | Test |
| ---------- | --------------: | ---: |
| **Baseline, GRU 128, no dropout** | 48.42% | **45.45%** |
| Dropout 0.3, GRU 128 | 48.73% | 44.89% |
| BatchNorm | 37.97% | 35.51% |
| **GRU 256, no dropout** | **49.37%** | 43.75% |

**Findings** 
- Adding dropout did not improve test accuracy.
- Increasing the GRU hidden size improved validation accuracy slightly but reduced test accuracy. 
- Adding BatchNorm in this configuration reduced both validation and test performance.
- The baseline configuration therefore remained the reference model for the next experiment. 

## 6.5 Experiment 3: Increasing Training Length 
**Objective:** Investigate whether training from 10 to 20 epochs gives the model more time to learn.

| Configuration | Epochs | Best validation | Test |
| ------------- | -----: | --------------: | ---: |
| Dropout | 10 | 48.73% | 44.89% |
| No dropout | 10 | 48.42% | 45.45% |
| No dropout | 20 | **52.85%** | **49.72%** |

**Findings** 
- Increasing training from 10 to 20 epochs improved both the best validation accuracy and test accuracy.
- Best result so far. However, the model was still overfitting with training accuracy continuing to rise much faster than validation accuracy

### 6.6 Experiment 4: Does the GRU Help? 
**Objective:**  Determine whether explicitly modelling temporal information with a GRU actually improved performance compared with using the CNN alone.

| Model | Best validation | Test |
| ----- | --------------: | ---: |
| CNN + GRU, 16 frames | 48.42% | **47.44%** |
| CNN only, 16 frames | **49.68%** | 43.18% |
| CNN + GRU, 32 frames | 49.37% | 41.76% |

**Findings** 
- Adding the GRU improved test accuracy by about 4 points over the CNN-only variant, and going from 16 to 32 frames did not help.

## 6.7 Experiment 5: Sequence Length — 8 vs 16 Frames
**Objective:**  Investigate whether the number of frames supplied to the temporal model affected generalisation.

| Metric               |  16 frames |     8 frames |
| -------------------- | ---------: | -----------: |
| Best validation      | **50.63%** |       48.73% |
| Test accuracy        |     48.01% |   **50.28%** |
| Final train accuracy |     98.15% |       97.66% |
| Train → test gap     |   50.14 pp | **47.38 pp** |

**Findings** 
- Using 8 frames produced the highest test accuracy of the video experiments at 50.28%.
- Shorter sequence also slightly reduced the training-to-test gap.

## 6.7 Experiment 5: Sequence Length — 8 vs 16 Frames
**Objective:** The baseline model showed severe overfitting so the goal is to investigate whether a different architecture could reduce the large fully connected CNN component and allow the temporal model to use information from the entire sequence.

### What changed
| Area                     | Version 1                                            | Version 2                                         | Reason                                                       |
| ------------------------ | ---------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------ |
| CNN head                 | Flatten 65,536 values → fully connected layer of 256 | **Global average pooling** directly to 256 values | Reduces the number of parameters substantially               |
| CNN blocks               | Conv + ReLU                                          | Conv + **BatchNorm + LeakyReLU**                  | More stable feature learning                                 |
| CNN pooling              | 3 max-pools                                          | 3 max-pools                                       | Unchanged                                                    |
| GRU direction            | Unidirectional                                       | **Bidirectional**                                 | Allows information from both directions through the sequence |
| Sequence summary         | Last hidden state                                    | **Additive attention**                            | Allows the model to weight different frames                  |
| Regularisation           | None                                                 | **Dropout 0.3**                                   | Reduce overfitting                                           |
| Classifier               | Linear, 128 → 7                                      | Linear, 256 → 7                                   | Matches bidirectional GRU output                             |
| Optimiser                | Adam                                                 | **AdamW**                                         | Adds weight decay                                            |
| Learning-rate scheduling | None                                                 | **ReduceLROnPlateau**                             | Reduce learning rate when validation loss stops improving    |
| Augmentation             | Off                                                  | Optional                                          | Tested separately                                            |

### Architecture
```
16 frames → CNN (4 blocks: Conv → BatchNorm → LeakyReLU, 3 max-pools, global average pool) → 256-D per frame
          → Bidirectional GRU (hidden 128 each way) → 256-D per time step
          → Attention (Linear 256 → 1, softmax over the 16 frames) → weighted sum → 256-D
          → Dropout 0.3 → Linear 256 → 7 → emotion logits
```

## 6.9 Experiment 7: Data Augmentation for Video 
**Objective:**  Investigate whether data augmentation improves generalisation.

| Configuration   | Train accuracy | Validation accuracy | Test accuracy |
| --------------- | -------------: | ------------------: | ------------: |
| No augmentation |         98.15% |              50.63% |    **48.01%** |
| + Augmentation  |         64.35% |          **55.70%** |        42.33% |

**Findings**
- Reduced training accuracy, indicating that the model struggles to memorise the training data. 
- Validation accuracy increased from 50.63% → 55.70%
- Test accuracy decreased from 48.01% → 42.33% 
- Did not produce a consistent improvement. 

## 6.10 Experiment 8: Final CNN + GRU Evaluation 
### Overall Results. 
| Split           |   Accuracy |
| --------------- | ---------: |
| Final training  |     98.15% |
| Best validation |     50.63% |
| Test            | **48.01%** |

## Training Progress: 
| Epoch | Train accuracy | Validation accuracy |
| ----: | -------------: | ------------------: |
|     1 |         19.11% |              17.72% |
|     5 |         64.70% |              38.92% |
|    10 |         89.70% |              43.67% |
|    15 |         96.24% |              42.09% |
|    20 |         98.15% |          **50.63%** |

### Test performance by emotion.
| Emotion   | Correct | Total | Accuracy |
| --------- | ------: | ----: | -------: |
| Anger     |      43 |    64 |   67.19% |
| Neutral   |      20 |    32 |   62.50% |
| Happiness |      38 |    64 |   59.38% |
| Fear      |      37 |    64 |   57.81% |
| Sadness   |      24 |    64 |   37.50% |
| Surprise  |       6 |    32 |   18.75% |
| Disgust   |       1 |    32 |    3.12% |

**Findings** 
- The model achieved a final training accuracy of 98.15%, but only 48.01% test accuracy.
- The large gap between training and unseen data performance indicates substantial overfitting.
- Performance also varied considerably between emotion classes. Anger, Neutral, Happiness, and Fear were recognised more frequently, while Surprise and particularly Disgust were difficult for the model. 

### 6.11 Overall Conclusion for Model 2 
- The best video experiments achieved approximately 48–50% test accuracy. 
- The main issue was overfitting. Model could achieve high training accuracy while performing substantially worse on unseen actors.
- Results suggest that increasing architectural complexity did not solve the underlying problem. The relatively small number of training videos and actors, combined with the actor-independent split, limited the model's ability to generalise. 

---
# 7. Return to Model 1: Expanded Dataset
## 7.1 Motivation
After experimenting with the video model, I began to question whether the additional temporal modelling was actually necessary for the final application. The video model introduced additional complexity and required a much smaller video dataset. Its performance was also substantially below the static-image CNN. I therefore decided to return to the original static-image approach and investigate whether improving the dataset could produce better results using a simpler and more established approach for emotion recognition.

AffectNet is especially useful for the weakest classes. Its test split has 305 Fear and 321 Disgust images, compared with 74 and 160 in the RAF-DB test split.

## 7.2 Dataset
[AffectNet YOLO Format Dataset](https://www.kaggle.com/datasets/fatihkgg/affectnet-yolo-format/data) (Kaggle). It is organised for object detection, with a label file per image, so it had to be converted to the folder-per-class layout used for the RAF-DB classifier.

## 7.3 Preprocessing pipeline
I wrote a preprocessing script that prepares AffectNet to match the RAF-DB CNN:
1. **Remove Contempt.** AffectNet has an eighth class (Contempt) that RAF-DB does not, so those images are skipped.
2. **Remap class IDs** to the project's seven-class order (below).
3. **Reorganise into `split/class/` folders** (the format used by PyTorch's `ImageFolder`), moving each image into the folder for its mapped class.
4. **Remove the old YOLO structure** (label files and emptied image folders).
5. **Check for duplicate images across splits** by hashing the decoded pixel values of every image (SHA-256) and flagging any hash that appears in more than one of train, validation, and test.
6. **Print the final class distribution** for each split to verify the result.

### Removing duplicate leaks
Duplicate images that appear in both a training split and an evaluation split would let the model "memorise" test images and inflate its scores. The check found **107 duplicates**, which were removed from the validation and test splits so the training data was left untouched.

| Split | Before | After | Removed |
| ----- | -----: | ----: | ------: |
| Train | 17,101 | 17,101 | 0 |
| Validation | 5,406 | 5,335 | 71 |
| Test | 2,755 | 2,719 | 36 |


## 7.4 Combined dataset
| Split | RAF-DB | AffectNet | Combined |
| ----- | -----: | --------: | -------: |
| Training pool | 12,271 | 15,105 | 27,376 |
| → Train | | | **24,639** |
| → Validation | | | **2,737** |
| Test | 3,068 | 2,389 | **5,457** |

Validation images are held out from the pooled training data (about 10%). The RAF-DB and AffectNet test sets are kept separate, so the model can be evaluated on each dataset individually as well as on the combined set.

## 7.5 Training
The static-image CNN was trained for **50 epochs** on the combined training set (770 batches of 32 per epoch). After each epoch the model was evaluated on the validation set, and the checkpoint with the best validation accuracy was saved (`emotion_cnn_raf_affectnet.pth`) and used for the final test.

| Epoch | Train loss | Train acc. | Validation acc. |
| ----: | ---------: | ---------: | --------------: |
| 1 | 1.7081 | 33.54% | 44.39% |
| 5 | 1.1199 | 57.87% | 61.42% |
| 10 | 0.9871 | 63.18% | 65.36% |
| 20 | 0.8771 | 67.29% | 66.35% |
| 30 | 0.8159 | 69.75% | 68.54% |
| 40 | 0.7888 | 70.78% | 68.98% |
| **47** | 0.7749 | 71.31% | **70.44% (best)** |
| 50 | 0.7586 | 71.80% | 70.11% |

Training and validation accuracy stay close together for the whole run (about 71.8% versus 70.1%), so unlike the video model this one is **not overfitting**. Both curves were still creeping upward at epoch 50, which suggests the model is limited by capacity or training time rather than by memorisation.

## 7.6 Results
Best checkpoint (validation accuracy 70.44%), evaluated on each test set:
| Test set | Images | Accuracy |
| -------- | -----: | -------: |
| RAF-DB only | 3,068 | **79.30%** |
| AffectNet only | 2,389 | 67.85% |
| **Combined** | 5,457 | **74.29%** |

AffectNet is a harder, more varied "in-the-wild" dataset, so lower accuracy there is expected.

Per-class recall:
| Emotion | RAF-DB test | AffectNet test | Combined test |
| ------- | ----------: | -------------: | ------------: |
| Surprise | 259 / 329 (78.72%) | 318 / 464 (68.53%) | 577 / 793 (72.76%) |
| Fear | 38 / 74 (51.35%) | 218 / 305 (71.48%) | 256 / 379 (67.55%) |
| Disgust | 57 / 160 (35.62%) | 191 / 321 (59.50%) | 248 / 481 (51.56%) |
| Happiness | 1,098 / 1,185 (92.66%) | 357 / 399 (89.47%) | 1,455 / 1,584 (91.86%) |
| Sadness | 364 / 478 (76.15%) | 135 / 276 (48.91%) | 499 / 754 (66.18%) |
| Anger | 101 / 162 (62.35%) | 246 / 374 (65.78%) | 347 / 536 (64.74%) |
| Neutral | 516 / 680 (75.88%) | 156 / 250 (62.40%) | 672 / 930 (72.26%) |

### Did adding AffectNet help? (like-for-like on the RAF-DB test set)
| Emotion | RAF-DB-only model | Trained on RAF-DB + AffectNet | Change |
| ------- | ----------------: | ----------------------------: | -----: |
| Surprise | 79.6% | 78.72% | -0.9 |
| Fear | 51.4% | 51.35% | 0.0 |
| Disgust | 30.6% | 35.62% | **+5.0** |
| Happiness | 90.7% | 92.66% | +2.0 |
| Sadness | 72.0% | 76.15% | **+4.2** |
| Anger | 64.2% | 62.35% | -1.9 |
| Neutral | 74.9% | 75.88% | +1.0 |
| **Overall** | **77.61%** | **79.30%** | **+1.7** |

### Findings
- Adding AffectNet raised RAF-DB test accuracy from **77.61% to 79.30%**, the best result of the project so far.
- The largest gains were on the weakest and least-represented classes: Disgust (+5.0 points, 8 more images correct) and Sadness (+4.2 points).
- Fear recall on RAF-DB did not change, and Surprise and Anger dipped slightly.
**Best result to date: 79.30% test accuracy on RAF-DB** (deeper CNN trained on RAF-DB + AffectNet), up from 77.61% with RAF-DB alone.