import os
import kagglehub
import pandas as pd

def main(): 
    # path = kagglehub.dataset_download("shuvoalok/raf-db-dataset")
    # print(f"Dataset location: \n {path} \n")
    path = "/Users/laurenpalega/.cache/kagglehub/datasets/shuvoalok/raf-db-dataset/versions/2"

    #reminder: once done delete cache rm -rf ~/.cache/kagglehub 

    #Folder structure -> OUTPUT: ['train_labels.csv', 'test_labels.csv', 'DATASET'] 
    print("\nTop level files:")
    print(os.listdir(path))

    # Contents of DATASET -> OUTPUT: ['test', 'train']
    dataset_folder = os.path.join(path, "DATASET")
    print(f"Contents of folder \n{os.listdir(dataset_folder)}\n")

    # Contents of DATASET/test -> OUTPUT: ['7', '6', '1', '4', '3', '2', '5']
    #1: Surprise 2:Fear 3:Disgust 4:Happiness 5:Sadness 6:Anger 7:Neutral 
    test_folder = os.path.join(path, "DATASET", "train")
    print(f"First 10 items of 'train'\n{os.listdir(test_folder)}\n")

    #Contents of DATASET/test/1 -> OUTPUT: ['train_04651_aligned.jpg', 'train_01420_aligned.jpg', 'train_08067_aligned.jpg', 'train_07072_aligned.jpg', 'train_07522_aligned.jpg']
    test1 = os.path.join(path, "DATASET", "train", "1")
    print(f"First 10 items of train \n{os.listdir(test1)[:5]}\n")

if __name__ == "__main__": 
    main() 