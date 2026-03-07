import os

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# dataset
DATASET_PATH = os.path.join(_BASE_DIR, "data")
DATASET_DISTRIBUTION = [0.7, 0.15, 0.15]
SEED = 42

LEARNING_RATE = 0.005
BATCH_SIZE = 8
EPOCH_NUMBER = 20
INPUT_SHAPE = 784

RESULT_PATH = os.path.join(_BASE_DIR, "results")

# Models
FINETUNING_MODEL = 'resnet50_20260307_115926_epoch20.pth'
CNN_MODEL = 'TODO'