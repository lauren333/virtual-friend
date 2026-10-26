""" 
Dataset preparation and loading for the emotion classification model. 
Provides preprocessed training and testing DataLoaders for model training and evaluation. 
""" 

from torchvision import datasets, transforms # Assist reaching/modyifing photos  
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler # Load data in batches for cnn 
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

# # --- Combine datasets ---
# train_dataset = ConcatDataset([ raf_train, affectnet_train ]) 
# test_dataset = ConcatDataset([ raf_test, affectnet_test ])

# # --- Carve a validation set out of TRAIN only ---
# # Test set stays completely untouched - we only use this val split
# # to decide things like epoch count / which checkpoint to keep.
# val_fraction = 0.1
# val_size = int(val_fraction * len(train_dataset))
# train_size = len(train_dataset) - val_size

# train_dataset, val_dataset = random_split(
#     train_dataset,
#     [train_size, val_size],
#     generator=torch.Generator().manual_seed(42)  # reproducible split
# )

# # --- Loading Training Dataset Batches ---
# train_loader = DataLoader(
#     train_dataset, 
#     batch_size = BATCH_SIZE, # Process 32 images at a time -> Input 4D tensor (32, 3, 128, 128)
#     shuffle=True # Randomize order of training images 
# ) 

# # --- Loading Testing Dataset Batches ---
# # Used to test model
# test_loader = DataLoader(
#     test_dataset, 
#     batch_size = BATCH_SIZE, # Process 32 images at a time -> Input 4D tensor  (32, 3, 128, 128) 
#     shuffle=False  # No need for testing
# ) 

# raf_test_loader = DataLoader( raf_test, batch_size=BATCH_SIZE, shuffle=False )
# affectnet_test_loader = DataLoader( affectnet_test, batch_size=BATCH_SIZE, shuffle=False )
# val_loader = DataLoader( val_dataset, batch_size=BATCH_SIZE, shuffle=False )

# --- Combine training datasets ---
# Training dataset uses augmentation.
raf_train = datasets.ImageFolder(
    root=f"{RAF_PATH}/train",
    transform=train_transform
)

affectnet_train = datasets.ImageFolder(
    root=f"{AFFECTNET_PATH}/train",
    transform=train_transform
)

train_dataset_full = ConcatDataset([
    raf_train,
    affectnet_train
])


# --- Create validation datasets ---
# Validation uses the same training images/split,
# but WITHOUT random augmentation.
raf_val = datasets.ImageFolder(
    root=f"{RAF_PATH}/train",
    transform=test_transform
)

affectnet_val = datasets.ImageFolder(
    root=f"{AFFECTNET_PATH}/train",
    transform=test_transform
)

val_dataset_full = ConcatDataset([
    raf_val,
    affectnet_val
])

# --- Check exact index mapping of classes ---
print("RAF-DB mapping:", raf_train.class_to_idx)
print("AffectNet mapping:", affectnet_train.class_to_idx)


# --- Create reproducible train/validation split ---
# Validation is taken ONLY from the training data.
# The test datasets remain completely untouched.
val_size = int(VAL_FRACTION * len(train_dataset_full))
train_size = len(train_dataset_full) - val_size

generator = torch.Generator().manual_seed(RANDOM_SEED)

indices = torch.randperm(
    len(train_dataset_full),
    generator=generator
)

train_indices = indices[:train_size]
val_indices = indices[train_size:]

### NEWWWWWWWWW
# --- Build sample weights for oversampling minority classes ---
# Get the class label for every image in the full training pool (RAF-DB + AffectNet, in order)
all_targets = raf_train.targets + affectnet_train.targets

# Pull out just the labels for the images that landed in train_dataset (not validation)
train_targets = [all_targets[i] for i in train_indices.tolist()]

# Inverse-frequency weight per class: rarer classes get higher weight
class_counts = torch.bincount(torch.tensor(train_targets))
class_weights = 1.0 / class_counts.float()
sample_weights = [class_weights[t] for t in train_targets]

# Sampler draws minority-class images more often, with replacement
train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)
print("Original train class counts:", class_counts.tolist())
### NEWWWWWWWWW

# --- Apply the same indices to both dataset versions ---
# train_dataset -> augmented images
# val_dataset   -> non-augmented images
train_dataset = torch.utils.data.Subset(
    train_dataset_full,
    train_indices
)

val_dataset = torch.utils.data.Subset(
    val_dataset_full,
    val_indices
)


# --- Combine testing datasets ---
# Test data is NEVER used for training or validation.
test_dataset = ConcatDataset([
    raf_test,
    affectnet_test
])


# --- Loading Training Dataset Batches ---
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True
# )
### NEWWWWWWWWW PREVIOUS ABOCE
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler   # replaces shuffle=True — sampler handles ordering + oversampling
)
### NEWWWWWWWWW


# --- Loading Validation Dataset Batches ---
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --- Loading Combined Testing Dataset ---
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --- Dataset-specific test loaders ---
raf_test_loader = DataLoader(
    raf_test,
    batch_size=BATCH_SIZE,
    shuffle=False
)

affectnet_test_loader = DataLoader(
    affectnet_test,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --- Dataset sizes ---
print(f"RAF-DB train:       {len(raf_train)}")
print(f"AffectNet train:    {len(affectnet_train)}")
print(f"Combined train:     {len(train_dataset)}")
print(f"Validation:         {len(val_dataset)}")
print(f"RAF-DB test:        {len(raf_test)}")
print(f"AffectNet test:     {len(affectnet_test)}")
print(f"Combined test:      {len(test_dataset)}")