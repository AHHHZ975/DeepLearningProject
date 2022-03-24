# Directories
ROOT_DIR = 'C:/Users/AmirHossein/OneDrive/Desktop/DeepLearningProject/DeepLearningProject'
DATA_DIR = ROOT_DIR + '/Data'
SHAPENET_DIR = DATA_DIR + '/ShapeNet'


# Image (x) parameters
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT


# Pointcloud (y) parameters
SAMPLE_SIZE = 1024


# Define training hyperparameters
INIT_LR = 0.00005
BATCH_SIZE = 5
EPOCHS = 95

# Define the train and val splits
TRAIN_SPLIT = 0.94
VAL_SPLIT = 1 - TRAIN_SPLIT
