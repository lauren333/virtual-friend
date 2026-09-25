"""
Prepare cached AffectNet data for the RAF-DB CNN.

Removes Contempt, maps AffectNet emotions to RAF-DB class indices,
reorganizes images into ImageFolder format, and verifies splits.
"""

from pathlib import Path
from collections import Counter
from PIL import Image
import hashlib
import shutil

# Configuration
from config import AFFECTNET_PATH
ROOT = Path(AFFECTNET_PATH)
SPLITS = ["train", "valid", "test"]
# AffectNet -> RAF-DB
CLASS_MAP = {
    0: 5,  # Anger
    2: 2,  # Disgust
    3: 1,  # Fear
    4: 3,  # Happy
    5: 6,  # Neutral
    6: 4,  # Sad
    7: 0,  # Surprise
}
# Original mapping of emotions in RAFDB
# Match exisiting structure. 
EMOTIONS = {
    0: "Surprise",
    1: "Fear",
    2: "Disgust",
    3: "Happiness",
    4: "Sadness",
    5: "Anger",
    6: "Neutral",
}
IMAGE_EXTENSIONS = { ".jpg", ".jpeg", ".png", ".bmp", ".webp" }

def image_hash(path):
    """Return a hash of the actual image pixels."""
    try:
        with Image.open(path) as image:
            image = image.convert("RGB") # Convert every image to RGB.
            # converts all of the pixel values into raw byte. 
            # hashlib.sha256(...) then creates a fingerprint of those pixels. 
            return hashlib.sha256(image.tobytes()).hexdigest() 
    except Exception:
        return None

def find_image(image_dir, stem):
    """Find the image matching a label filename."""
    # Go through every image extension  
    for ext in IMAGE_EXTENSIONS:
        # creates: ".../train/images/image123.jpg"
        path = image_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None

def get_class(label_path):
    """Return the first valid AffectNet class in a YOLO label."""
    with open(label_path, "r") as file:
        for line in file: # Read the label file one line at a time.
            parts = line.strip().split()
            if not parts:
                continue
            try:
                # The first value in a YOLO annotation is the class ID.
                class_id = int(parts[0])
            except ValueError:
                continue
            if class_id in CLASS_MAP:
                return class_id
    return None

def convert():
    """Convert dataset."""
    print("CONVERTING AFFECTNET CACHE TO RAF-DB CLASS MAPPING")
    print("=" * 70)
    print(f"\nDataset: {ROOT}")
    # Process train, validation and test separately.
    for split in SPLITS:
        image_dir = ROOT / split / "images"
        label_dir = ROOT / split / "labels"
        # Make sure both directories exist before processing them.
        if not image_dir.exists() or not label_dir.exists():
            print(f"\nWARNING: Missing {split} images or labels")
            continue
        print(f"\nProcessing {split.upper()}...")
        counts = Counter()
        for label_path in label_dir.glob("*.txt"):
            # Read the AffectNet class from the label file.
            source_class = get_class(label_path)
            # Class 1 = Contempt, so skip it.
            if source_class is None:
                continue
            # Convert the AffectNet class ID into the RAF-DB class id
            target_class = CLASS_MAP[source_class]
            # Find the actual image that belongs to this label.
            image_path = find_image(image_dir, label_path.stem)
            if image_path is None:
                continue
            # Create the destination folder for the RAF-DB class.
            target_dir = ROOT / split / str(target_class)
            target_dir.mkdir(exist_ok=True)
            # Create the final destination path for the image.
            target_path = target_dir / image_path.name
            # Reorganises the dataset.
            shutil.move( image_path, target_path)
            # Increase the count for this emotion.
            counts[target_class] += 1
        # Print the resulting number of images in each class.
        print("\nFinal classes:")
        for class_id, emotion in EMOTIONS.items():
            print(
                f"{class_id}: "
                f"{emotion:<10} "
                f"{counts[class_id]:>6}"
            )

def remove_old_structure():
    """Remove old YOLO labels/directories"""
    for split in SPLITS:
        labels = ROOT / split / "labels"
        # If it exists, delete the entire directory.
        if labels.exists():
            shutil.rmtree(labels)
        # Locate the old images directory.
        images = ROOT / split / "images"
        # After conversion, this directory should be empty because all valid images were moved into class folders.
        if images.exists() and not any(images.iterdir()):
            images.rmdir() # Remove the now-empty directory.


def check_duplicates():
    """ Check for exact duplicate images across train/validation/test. """ 
    print("FINAL CROSS-SPLIT DUPLICATE CHECK")
    print("=" * 70)
    hashes = {}
    # Process train, validation and test.
    for split in SPLITS:
        split_dir = ROOT / split
        # Look through everything inside the split directory.
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            # Look through the files inside the class directory.
            for image_path in class_dir.iterdir():
                # Ignore anything that isn't a supported image.
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                # Calculate the hash of the image's actual pixels.
                h = image_hash(image_path)
                if h is None:
                    continue
                # Add the image location to the list associated with its hash.
                hashes.setdefault(h, []).append( (split, image_path) )
    duplicates = []
    # Look through every group of identical image hashes.
    for entries in hashes.values():
        # means the same image exists in both splits.
        splits = {split for split, _ in entries}
        # We only care about duplicates that cross dataset splits, A duplicate within the same split is not train/test leakage.
        if len(splits) > 1:
            duplicates.append(entries)
    if duplicates:
        # Print how many duplicate groups remain.
        print(
            f"\nWARNING: {len(duplicates)} "
            "cross-split duplicates remain."
        )
        for entries in duplicates[:10]:
            print()
            # Show where each copy of the image was found.
            for split, path in entries:
                print(f"{split.upper()}: {path}")
    else:
        print("\nNo exact pixel-level duplicates remain.")


def print_distribution():
    """ Print the final number and percentage of images in each class. """ 
    print("FINAL EMOTION DISTRIBUTION")
    print("=" * 70)
    for split in SPLITS:
        # Counter stores the number of images in each class.
        counts = Counter()
        split_dir = ROOT / split
        for class_id in EMOTIONS:
            class_dir = split_dir / str(class_id)
            if class_dir.exists():
                # Count files with supported image extensions.
                counts[class_id] = sum(
                    1
                    for p in class_dir.iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS
                )
        # Calculate the total number of images in this split.
        total = sum(counts.values())
        print(f"\n{split.upper()}")
        print("-" * 70)
        # Print each class and its percentage of the split.
        for class_id, emotion in EMOTIONS.items():
            count = counts[class_id]
            percentage = (
                100 * count / total
                if total
                else 0
            )
            print(
                f"{class_id}: "
                f"{emotion:<10} "
                f"{count:>6} "
                f"({percentage:>6.2f}%)"
            )
        print("-" * 70)
        print(f"TOTAL: {total}")

# Main
def main():
    convert()
    remove_old_structure()
    print_distribution()
    check_duplicates()
    print()
    print("DONE")

# Script entry point
if __name__ == "__main__":
    main()