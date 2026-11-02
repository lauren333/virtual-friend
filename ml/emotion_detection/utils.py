import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from ml.emotion_detection.config import EMOTION_NAMES, NUM_CLASSES

def get_device():
    """ 
        Return MPS when available, otherwise use CPU. 
        Apple M4 Macs can use Apple's Metal Performance Shaders (MPS)
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def plot_confusion_matrix(labels, predictions, title):
    """
    Plot a confusion matrix using the project's emotion classes.
    """
    cm = confusion_matrix(labels, predictions, labels=list(range(NUM_CLASSES)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=EMOTION_NAMES)
    disp.plot(xticks_rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.show()