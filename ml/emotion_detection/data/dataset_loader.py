""" 
Dataset preparation and loading for the emotion classification model. 
Provides preprocessed training and testing DataLoaders for model training and evaluation. 
""" 

from torchvision import datasets, transforms # Assist reaching/modyifing photos  
from torch.utils.data import DataLoader, ConcatDataset, random_split # Load data in batches for cnn 
import torch
from ml.emotion_detection.config import BATCH_SIZE, IMAGE_SIZE, RAF_PATH, AFFECTNET_PATH, VAL_FRACTION, RANDOM_SEED 
# Current configation: Batch Size: 32, Image Size: 128

# --- Training preprocessing ---
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2, 
        saturation=0.15
    ),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- Testing preprocessing ---
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- Preprocessing Training Dataset ---
raf_train = datasets.ImageFolder(
    root = f"{RAF_PATH}/train", # seven emotion classes 
    transform = train_transform # Apply transformations / loaded image 
)
affectnet_train = datasets.ImageFolder( 
    root=f"{AFFECTNET_PATH}/train", 
    transform=train_transform 
)

# --- Preprocessing Testing Dataset ---
raf_test = datasets.ImageFolder(
    root = f"{RAF_PATH}/test",  
    transform = test_transform 
)
affectnet_test = datasets.ImageFolder( 
    root=f"{AFFECTNET_PATH}/test", 
    transform=test_transform 
)

# --- Check exact index mapping of classes ---
print("RAF-DB mapping:", raf_train.class_to_idx) 
print("AffectNet mapping:", affectnet_train.class_to_idx)

# --- Combine datasets ---
train_dataset = ConcatDataset([ raf_train, affectnet_train ]) 
test_dataset = ConcatDataset([ raf_test, affectnet_test ])

# --- Carve a validation set out of TRAIN only ---
# Test set stays completely untouched - we only use this val split
# to decide things like epoch count / which checkpoint to keep.
val_fraction = 0.1
val_size = int(val_fraction * len(train_dataset))
train_size = len(train_dataset) - val_size

train_dataset, val_dataset = random_split(
    train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # reproducible split
)

# --- Loading Training Dataset Batches ---
train_loader = DataLoader(
    train_dataset, 
    batch_size = BATCH_SIZE, # Process 32 images at a time -> Input 4D tensor (32, 3, 128, 128)
    shuffle=True # Randomize order of training images 
) 

# --- Loading Testing Dataset Batches ---
# Used to test model
test_loader = DataLoader(
    test_dataset, 
    batch_size = BATCH_SIZE, # Process 32 images at a time -> Input 4D tensor  (32, 3, 128, 128) 
    shuffle=False  # No need for testing
) 

raf_test_loader = DataLoader( raf_test, batch_size=BATCH_SIZE, shuffle=False )
affectnet_test_loader = DataLoader( affectnet_test, batch_size=BATCH_SIZE, shuffle=False )
val_loader = DataLoader( val_dataset, batch_size=BATCH_SIZE, shuffle=False )

print(f"RAF-DB train:       {len(raf_train)}")
print(f"AffectNet train:    {len(affectnet_train)}")
print(f"Combined train:     {len(train_dataset)}")
print(f"Validation:         {len(val_dataset)}")
print(f"RAF-DB test:        {len(raf_test)}")
print(f"AffectNet test:     {len(affectnet_test)}")
print(f"Combined test:      {len(test_dataset)}")