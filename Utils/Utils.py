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


def imageToPatches(inmage, patchSize, isFlattenChannels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patchSize, patchSize, W//patchSize, patchSize)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)              # [B, H'*W', C, p_H, p_W]
    if isFlattenChannels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    return x