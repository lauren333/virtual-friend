# --- Dataset --- 
RAF_PATH = "/Users/laurenpalega/.cache/kagglehub/datasets/shuvoalok/raf-db-dataset/versions/2/DATASET"
AFFECTNET_PATH = "/Users/laurenpalega/.cache/kagglehub/datasets/fatihkgg/affectnet-yolo-format/versions/2/YOLO_format"

# --- Dataset Split ---
VAL_FRACTION = 0.1
RANDOM_SEED = 42

# --- Model --- 
NUM_CLASSES = 7
EMOTION_NAMES = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral",]

# --- Training ---
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32

# --- Image preprocessing --- 
IMAGE_SIZE = 128

# --- Model checkpoints --- 
BEST_MODEL_PATH = "/Users/laurenpalega/Documents/virtualfriend/ml/emotion_detection/checkpoints/emotion_cnn_raf_affectnet_best.pth"
BEST_MODEL_ONLY_RAFDB = "/Users/laurenpalega/Documents/virtualfriend/ml/emotion_detection/checkpoints/emotion_cnn_rafdb_best.pth"