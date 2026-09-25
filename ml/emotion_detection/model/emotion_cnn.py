""" 
Convolutional neural network for facial expression classification.
Defines the EmotionCNN architecture used to extract visual features 
and classify input images into seven emotion classes. 
"""

import torch
import torch.nn as nn
import torchvision 

class EmotionCNN(nn.Module): 
    def __init__(self):
        super().__init__()

        # 1. FEATURE EXTRACTION  ------

        # --- Convolution Layer 1 --- #
        # GOAL 1: Learn local patterns with convolutions 
        # Input: H x W x 3 (RGB image) -> Output: 32 feature maps
        self.conv1 = nn.Conv2d(
            # RGB Scale has 3 color channeñs 
            in_channels = 3, 
            # How many features/filters should it learn 
            out_channels = 32,
            # Size of kernel to slide across image -> 3x3 
            kernel_size = 3,
            # Kernel slides/steps 1px 
            stride = 1, 
            # Padding to input image -> new image/ matrix with 0s as border   
            padding = 1
        )
        # Activation Function -> performs f(x) = max(0,x) for output of conv1 -> put 0 if less than 0
        self.relu1 = nn.ReLU()
        # Pooling: 
        # GOAL 2: Trade spatial resolution for richer feature representation 
        self.pool1 = nn.MaxPool2d(
            # 2x2 kernel/slider 
            kernel_size = 2,
            # move pooling window 2 pixels -> non overlaping pooling regions -> decreases spacial resolution significantly
            stride = 2
        )

        # --- Convolution Layer 2 ---- #
        # INPUT: Output of conv1, 32 feature maps/channels -> OUTPUT: 64 feature maps
        # The same kernel, stride, padding, activation, and pooling 
        self.conv2 = nn.Conv2d(
            in_channels = 32, 
            out_channels = 64, 
            kernel_size = 3, 
            stride = 1, 
            padding = 1
        )
        # Activation  
        self.relu2 = nn.ReLU() 
        # Pooling  
        self.pool2 = nn.MaxPool2d( 
            kernel_size = 2,
            stride = 2
        )

        # --- Convolution Layer 3 ---- #
        # INPUT: Output of conv2, 64 feature maps/channels -> OUTPUT: 128 feature maps
        self.conv3 = nn.Conv2d(
            in_channels = 64, 
            out_channels = 128, 
            kernel_size = 3, 
            stride = 1, 
            padding = 1
        )
        # Activation 
        self.relu3 = nn.ReLU() 
        # No more pooling  

        # --- Convolution Layer 4 ---- #
        self.conv4 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            padding=1
        )

        self.relu4 = nn.ReLU()

        self.pool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        ) 


        # Notes: Notice the spatial resolution is decreasing and feature channels is increasing 
        # Imagine a 3d box the front facing peice is shrinking (Width and Height), as it is become longer wise (Length)
        # This again, trades image resolution for richer feature representations 

        # OUTPUT OF FEATURE EXTRACTION: ----- 
        # CNN finished extracting visual features. 
        # Current tensor: (batch_size, 128, 32, 32) -> batch size is # of images processed at once, 128 learned feature maps (channels), 32 px by 32px
        # Imagine 128 different 32x32 feature maps stacked together -> forming a 3d block -> each feature map has learned to detect a different visual pattern 

        # 2. CLASSIFIER ------
        # Converts feature maps into one long vector 
        self.flatten = nn.Flatten() 

        # After flattening -> 128 feature maps x 32 x 32 pixels = 131072 features per image -> Current Tensor now (batch_size, 131072)
        # Compress them to get first fully connected linear layer 
        # Each of the 256 neurons is connected to ALL 131072 input features -> each nueron learns 131072 wights and 1 bias
        # During training, backpropagation computes gradient of loss wrt every weight and bias 
        # Gradient descent then updates weights and bias to minimize cost function 
        # self.classifier1 = nn.Linear(
        #     in_features = 131072, 
        #     out_features= 256
        # )
        self.classifier1 = nn.Linear(
            in_features = 65536, 
            out_features= 256
        )

        # Activation 
        self.relu5 = nn.ReLU() 

        # Dropout to reduce overfitting
        self.dropout = nn.Dropout(0.3)

        # Final Classification Layer: 

        # INPUT: 256 learned feature activations -> OUTPUT: one logit for each emotion class -> ouput tensor (batch_size, 7) 
        # One output neuron per emotion (7 emotions from our training data)
        self.classifier2 = nn.Linear(
            in_features = 256, 
            out_features = 7
        ) 

      
    def forward(self,x):
        # x = input tensor (batch of images) 

        # 1. FEATURE EXTRACTION ------
        # Block 1 
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # print(f"After Block 1:: {x.shape}")

        # Block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # print(f"After Block 2: {x.shape}")

        # Block 3
        x = self.conv3(x)
        x = self.relu3(x)
        # print(f"After Block 3: {x.shape}")

        # Block 4
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool3(x)


        # 2. CLASSIFIER ------
        # Flatten feature maps into one long vector
        x = self.flatten(x)
        # print(f"After Flatten: {x.shape}")

        # First fully connected layer
        x = self.classifier1(x)
        x = self.relu5(x)
        # print(f"After Classifier 1: {x.shape}")

        # Dropout during training
        x = self.dropout(x)

        # Final classification layer
        x = self.classifier2(x)
        # print(f"Output: {x.shape}")

        #Returns the logits for the 7 emotion classes
        return x 