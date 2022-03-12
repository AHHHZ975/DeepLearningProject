import glob
import matplotlib.pyplot as plt

def getOBJsInDirectory(path):
    """
    This function recursively finds for all .obj
    files in a directory and return their paths.
    """
    objectPaths = []
    for filename in glob.iglob(path + '/**/*.obj', recursive = True):
        objectPaths.append(filename)
    return objectPaths

def getPLYsInDirectory(path):
    """
    This function recursively finds for all .ply
    files in a directory and return their paths.
    """
    plyPaths = []
    for filename in glob.iglob(path + '/**/*.ply', recursive = True):
        plyPaths.append(filename)
    return plyPaths

def getRGBsInDirectory(path):
    """
    This function recursively finds for all .jpg
    files in a directory and return their paths.
    """
    objectPaths = []
    for filename in glob.iglob(path + '/**/*.jpg', recursive = True):
        objectPaths.append(filename)
    return objectPaths

def save_loss_plot(H):
    """
    Saves and diplays loss and accuracies of train and validation data.
    """
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="Training")
    plt.plot(H["val_loss"], label="Validation")
    plt.plot(H["test_loss"], label="Testing")
    plt.title("Training/Validation/Test Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss (Chamfer Distance)")
    plt.legend(loc="upper right")
    plt.savefig('TrainResult.jpg')
    # plt.show()  