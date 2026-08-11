""" 
Dataset preparation and loading for the emotion classification model. 
Provides preprocessed training and testing DataLoaders for model training and evaluation. 
""" 

from torchvision import datasets, transforms # Assist reaching/modyifing photos  
from torch.utils.data import DataLoader # Load data in batches for cnn 

DATASET_PATH = "/Users/laurenpalega/.cache/kagglehub/datasets/shuvoalok/raf-db-dataset/versions/2/DATASET"

# --- Preprocessing --- 
transform = transforms.Compose([
    transforms.Resize((128,128)), # Resize image to 128px × 128px to match CNN architecture     
    transforms.ToTensor() # Convert HxWxC -> 2D Tensor CxHxW (3,128,128) and scale pixels [0,1] 
])

# --- Preprocessing Training Dataset ---
train_dataset = datasets.ImageFolder(
    root = f"{DATASET_PATH}/train", # seven emotion classes 
    transform = transform # Apply transformations / loaded image 
)
print("Emotion class mapping:", train_dataset.class_to_idx) # Check exact index mapping of classes


# --- Preprocessing Testing Dataset ---
test_dataset = datasets.ImageFolder(
    root = f"{DATASET_PATH}/test",  
    transform = transform 
)

# --- Loading Training Dataset Batches ---
train_loader = DataLoader(
    train_dataset, 
    batch_size = 32, # Process 32 images at a time -> Input 4D tensor (32, 3, 128, 128)
    shuffle=True # Randomize order of training images 
) 

# --- Loading Testing Dataset Batches ---
# Used to test model
test_loader = DataLoader(
    test_dataset, 
    batch_size = 32, # Process 32 images at a time -> Input 4D tensor  (32, 3, 128, 128) 
    shuffle=False  # No need for testing
) 