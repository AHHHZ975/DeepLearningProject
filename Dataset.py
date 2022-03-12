import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import Config as cfg
from Tools import Tool3D
import open3d as o3d
import os


# Setup device (CPU or GPU)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print("GPU\n")
    
else:
    device = torch.device("cpu")
    print("CPU")


class ShapeNetDataset(Dataset):
    """
    This is a cutomized dataset has been written in Pytorch for
    loading ShapeNet dataset as RGBs (x) and pointclouds (y).
    In the following functions, I just overloaded some required
    functions in Pytorch's datasets.
    """
    def __init__(self, numberOfData, train=True, demo=False, demoPath='None'):
        # Loading train/val/test data
        # self.samples = (x: images, y: pointclouds)
        self.samples = loadTrainData(numberOfData, train, demo, demoPath)

        # A transform will be used to convert my data to PyTorch tensor.
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        # The numeber of train data
        return len(self.samples)

    def __getitem__(self, idx):  
        # This just converts my input image and pointclouds to PyTorch tensor.
        return self.transform(self.samples[idx]['x']).float(), self.transform(self.samples[idx]['y']).float()

def loadRGB(path):
    """
    Loads RGB synthetic images which have already been generated
    using Open3D tool. This data will be used as train/train data (X)
    in the taining process.
    """
    # Loading images using OpenCV          
    return cv2.imread(path)

def loadPointCloud(path):
    """
    Loads pointclouds which have already been generated using
    Open3D tool. This data will be used as train data (x) in 
    the taining process.
    """
    Open3D_pointcloud = Tool3D.loadPointCloud(path)
    # Tool3D.visualize(Open3D_pointcloud)
    xyz_pointcloud = Tool3D.pointCloud2XYZ(Open3D_pointcloud)
    return xyz_pointcloud

def loadTrainData(numberOfData, train, demo, demoPath):
    """
    Loads the RGBs ("X" known as train data),
    and their corresponding pointclouds 
    ("Y" known as labels)
    """
    if demo:
        demoData = []
        imagePath, pointcloudPath = demoPath
        # # Load X (train data)
        image = loadRGB(imagePath)

        # Load Y (labels)
        pointCloud = loadPointCloud(pointcloudPath)

        # Packing X, Y as a dictionary        
        demoData.append({'x': image, 'y': pointCloud})
        return demoData

    else:
        # Variables
        trainData = []
        sourcePath = cfg.ROOT_DIR + '/Output/GeneratedData'

        # Create source and destination directory
        if train:
            sourcePath += '/Train'
        else:
            sourcePath += '/Test'

        immediatePaths = next(os.walk(sourcePath))[1]

        # Loading mesh objects that have been placed in the ShapeNet dataset directory
        
        for immediatePath in immediatePaths:
            imediateImediatePath = next(os.walk(f'{sourcePath}/{immediatePath}'))[1]
            imediateImediatePath = imediateImediatePath[0:numberOfData]
            for i in range(len(imediateImediatePath)):
                # # Load X (train data)
                imagePath = f'{sourcePath}/{immediatePath}/{i}/{i}.jpg'
                image = loadRGB(imagePath)

                # Load Y (labels)
                pointcloudPath = f'{sourcePath}/{immediatePath}/{i}/{i}.ply'
                pointCloud = loadPointCloud(pointcloudPath)
            
                # Packing X, Y as a dictionary        
                trainData.append({'x': image, 'y': pointCloud})
        
        return trainData

def displayTrainData(trainData):
    print(len(trainData))
    pointCloud = o3d.geometry.PointCloud()

    # Show te generated train data
    for i in range(len(trainData)):
        cv2.imshow('image', trainData[i]['x'])
        cv2.waitKey(1000)
        pointCloud.points = o3d.utility.Vector3dVector(trainData[i]['y'])
        o3d.visualization.draw_geometries([pointCloud],
                                            zoom=0.9, 
                                            front=[0.4, 0.5, -0.5],
                                            lookat=[0, 0.03, 0], 
                                            up=[0, 1, 0]
                                        )


if __name__ == '__main__':
    
    # Load train data
    trainData = loadTrainData(100, False)

    # Display train data
    displayTrainData(trainData)