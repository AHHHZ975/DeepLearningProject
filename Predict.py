# Set the numpy seed for better reproducibility
import numpy as np
from numpy.lib.type_check import imag
from torch.nn.modules import loss

from Engine import train
np.random.seed(42)

# import the necessary packages
from torch.utils.data import DataLoader     # Used to load my ShapeNet testing data
import torch
import Dataset
import sys
sys.path.append('/home/ahz/Desktop/3D-Reconstruction/3D-Reconstruction')
import Config as cfg
from torch.autograd import Variable
from Tools import Tool3D
import open3d as o3d
import time
from neuralnet_pytorch.metrics import chamfer_loss

numberOfTestData = 5000

# set the device I will be using to test the model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# Load the ShapeNet dataset
print("[INFO] loading the ShapeNet dataset...")
testData = Dataset.ShapeNetDataset(numberOfTestData, train=False)

# initialize the test data loader
testDataLoader = DataLoader(testData, batch_size=1)

# load the model and set it to evaluation mode
# modelPath = cfg.ROOT_DIR + '/PreTrainedModels/Ours/Airplane/Ours_3230TrainData_5BatchSize_30Epochs_0.94SPLIT_0.00005LR.pt'
# modelPath = cfg.ROOT_DIR + '/PreTrainedModels/Ours/Bench/Ours_1640TrainData_5BatchSize_35Epochs_0.94SPLIT_0.00005LR.pt'
# modelPath = cfg.ROOT_DIR + '/PreTrainedModels/Ours/Bottle/Ours_470TrainData_5BatchSize_50Epochs_0.94SPLIT_0.00005LR.pt'
# modelPath = cfg.ROOT_DIR + '/PreTrainedModels/Ours/Car/Ours_3163TrainData_5BatchSize_45Epochs_0.94SPLIT_0.00005LR.pt'
# modelPath = cfg.ROOT_DIR + '/PreTrainedModels/Ours/Cellphone/Ours_750TrainData_5BatchSize_45Epochs_0.94SPLIT_0.00005LR.pt'
# modelPath = cfg.ROOT_DIR + '/PreTrainedModels/Ours/Rifle/Ours_2140TrainData_5BatchSize_50Epochs_0.94SPLIT_0.00005LR.pt'
# modelPath = cfg.ROOT_DIR + '/PreTrainedModels/Ours/Sofa/Ours_2860TrainData_5BatchSize_40Epochs_0.94SPLIT_0.00005LR.pt'
# modelPath = cfg.ROOT_DIR + '/PreTrainedModels/Ours/Bike/Ours_304TrainData_5BatchSize_200Epochs_0.94SPLIT_0.00005LR.pt'


# modelPath = cfg.ROOT_DIR + '/PreTrainedModels/Ours_New/Airplane/Ours_3230TrainData_5BatchSize_50Epochs_0.94SPLIT_0.00005LR.pt'
# modelPath = cfg.ROOT_DIR + '/PreTrainedModels/Ours_New/Bench/Ours_1640TrainData_5BatchSize_50Epochs_0.94SPLIT_0.00005LR.pt'
# modelPath = cfg.ROOT_DIR + '/PreTrainedModels/Ours_New/Bottle/Ours_470TrainData_5BatchSize_60Epochs_0.94SPLIT_0.00005LR.pt'
# modelPath = cfg.ROOT_DIR + '/PreTrainedModels/Ours_New/Car/Ours_3163TrainData_5BatchSize_50Epochs_0.94SPLIT_0.00005LR.pt'
# modelPath = cfg.ROOT_DIR + '/PreTrainedModels/Ours_New/Cellphone/Ours_750TrainData_5BatchSize_70Epochs_0.94SPLIT_0.00005LR.pt'
# modelPath = cfg.ROOT_DIR + '/PreTrainedModels/Ours_New/Rifle/Ours_2140TrainData_5BatchSize_50Epochs_0.94SPLIT_0.00005LR.pt'
# modelPath = cfg.ROOT_DIR + '/PreTrainedModels/Ours_New/Sofa/Ours_2860TrainData_5BatchSize_50Epochs_0.94SPLIT_0.00005LR.pt'



# modelPath = cfg.ROOT_DIR + '/PreTrainedModels/PSGN/Airplane/PSGN_3230TrainData_5BatchSize_200Epochs_0.94SPLIT_0.0001LR.pt'
# modelPath = cfg.ROOT_DIR + '/PreTrainedModels/PSGN/Bottle/PSGN_470TrainData_5BatchSize_120Epochs_0.94SPLIT_0.00005LR.pt'
# modelPath = cfg.ROOT_DIR + '/PreTrainedModels/PSGN/Car/PSGN_2970TrainData_5BatchSize_100Epochs_0.94SPLIT_0.0001LR.pt'
# modelPath = cfg.ROOT_DIR + '/PreTrainedModels/PSGN/Rifle/PSGN_2140TrainData_5BatchSize_100Epochs_0.94SPLIT_0.00005LR.pt'


# modelPath = cfg.ROOT_DIR + '/PreTrainedModels/Pixel2Point/Bottle/Pixel2Point_470TrainData_5BatchSize_60Epochs_0.94SPLIT_0.00005LR.pt'
# modelPath = cfg.ROOT_DIR + '/PreTrainedModels/Pixel2Point/Rifle/Pixel2Point_2140TrainData_5BatchSize_60Epochs_0.94SPLIT_0.00005LR.pt'
# modelPath = cfg.ROOT_DIR + '/PreTrainedModels/Pixel2Point/Airplane/Pixel2Point_3230TrainData_5BatchSize_60Epochs_0.94SPLIT_0.00005LR.pt'
# modelPath = cfg.ROOT_DIR + '/PreTrainedModels/Pixel2Point_InitialPC/Bottle/Pixel2Point_470TrainData_5BatchSize_70Epochs_0.94SPLIT_0.00005LR.pt'


modelPath = cfg.ROOT_DIR + '/Convolution_AE.pt'

model = torch.load(modelPath).to(device)
model.eval()

# Chamfer distance as loss function
# chamferDist = ChamferDistance()

# switch off autograd
with torch.no_grad():
    
    # Initialize a list to store our predictions
    preds = []
    losses = []
    elapsedTimes = []

    # load over the test set
    for (x, y) in testDataLoader:
        # Reshaping the image  
        # x = x.view(x.size(0), -1)
        x = Variable(x)

        # Send the input to the device
        (x, y) = (x.to(device), y.to(device))

        # make the predictions and add them to the list
        startTime = time.time()
        # pred, mu, logvar = model(x)
        pred = model(x)
        endTime = time.time()
        elapsedTime = endTime - startTime
        
        # Calculate the training loss
        pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
        y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
        loss = chamfer_loss(pred, y)

        # Storing predictions and loss values
        elapsedTimes.append(elapsedTime)
        preds.append(pred)
        losses.append(loss)
    
    # Displaying the average loss during the evaluation of the model on the test data 
    print("Number of test data: ", len(losses))

    # Displaying the average loss during the evaluation of the model on the test data 
    print("Average loss: ", (sum(losses) / len(losses)).detach().cpu().numpy())

    # Displaying the average elapsed time during the evaluation of the model on the test data 
    print(f'[INFO] average time:{(sum(elapsedTimes) / len(elapsedTimes))*1000: .4f}ms')

    # Displaying the train data (X: RGB images) and prediction (Y: Pointclouds) together
    for i, pred in enumerate(preds):
        path = cfg.ROOT_DIR + '/Output/GeneratedData/Test/02876657'
        path += f'/{i}/{i}.jpg'
        image = Tool3D.loadImage(path)
        print(path)
        print("Chamfer loss:", losses[i].detach().cpu().numpy())
        pred = pred.view(cfg.SAMPLE_SIZE, 3).detach().cpu().numpy()
        pred = Tool3D.XYZ2PointCloud(pred)
        # pred = Tool3D.pcl2Voxel(pred, 0.02)
        o3d.visualization.draw_geometries([pred],
                                        zoom=0.9, 
                                        front=[0.5, 0.4, -0.6],
                                        lookat=[0, 0, 0], 
                                        up=[0, 1, 0]
                                        )
