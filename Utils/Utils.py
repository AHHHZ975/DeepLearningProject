import glob
import matplotlib.pyplot as plt
import torchvision

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


def imageToPatches(image, patchSize, isFlattenChannels=False):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = image.shape
    image = image.reshape(B, C, H//patchSize, patchSize, W//patchSize, patchSize)
    image = image.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    image = image.flatten(1, 2)              # [B, H'*W', C, p_H, p_W]
    if isFlattenChannels:
        image = image.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    return image


def showPatchedImage(img_patches, nrow=256):    

    fig, ax = plt.subplots(img_patches.shape[0], 1, figsize=(14,3))    
    fig.suptitle("Images as input sequences of patches")

    for i in range(img_patches.shape[0]):
        img_grid = torchvision.utils.make_grid(img_patches[0].float(), nrow=nrow, normalize=True, pad_value=0.9)        
        img_grid = img_grid.permute(1, 2, 0)        
        ax.imshow(img_grid)
        ax.axis('off')

    fig.savefig('PatchedImages.png', bbox_inches='tight')


