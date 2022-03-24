# set the matplotlib backend so figures can be saved in the background
import pickle
from torch.functional import Tensor
from torch.nn.modules import loss                                   # import matplotlib and set the appropriate background engine.
# matplotlib.use("TkAgg")
# matplotlib.use("Agg")

# import the necessary packages
from NN import AE, CAE_AHZ, PSGN, CAE_new, PSGN_Vanilla, Pixel2Point, Pixel2Point_InitialPC                                   # My PyTorch implementation of the simple autoencoder
from NN import CVAE                                  # My PyTorch implementation of the convolutional autoencoder
from torch.autograd import Variable
from torch.utils.data import random_split           # Constructs a random training/testing split from an input set of data
from torch.utils.data import DataLoader             # PyTorch’s awesome data loading utility that allows us to effortlessly build data pipelines to train our CNN
from torch.optim import Adam                        # The optimizer we’ll use to train our neural network
from torch.optim import SGD                        # The optimizer we’ll use to train our neural network
from torch import nn, utils                                # PyTorch’s neural network implementations
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import Dataset
from Tools import Tool3D
import sys
sys.path.append('C:/Users/AmirHossein/OneDrive/Desktop/DeepLearningProject/DeepLearningProject')
import Config as cfg
from neuralnet_pytorch.metrics import chamfer_loss
from Utils import Utils
# from swd import SinkhornDistance

def trainInitialization():
    numberOfTrainData = 470
    numberOfTestData = 25
    
    global device
    global trainDataLoader, valDataLoader, testDataLoader
    global trainSteps, valSteps, testSteps

    # Setup device (CPU or GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        print("GPU")
        
    else:
        device = torch.device("cpu")
        print("CPU")

    
    # Load the ShapeNet dataset
    print("[INFO] loading the ShapeNet dataset...")
    trainData = Dataset.ShapeNetDataset(numberOfTrainData, train=True)    
    testData = Dataset.ShapeNetDataset(numberOfTestData, train=False)


    # Calculate the train/validation split
    print("[INFO] generating the train/validation split...")
    numTrainSamples = int(len(trainData) * cfg.TRAIN_SPLIT)
    numValSamples = int(len(trainData) * cfg.VAL_SPLIT)
    numTestSamples = len(testData)

    valData = []
    splittedTrainData = []

    for i in range(numValSamples):
        valData.append(trainData[numTrainSamples + i])

    for i in range(numTrainSamples):
        splittedTrainData.append(trainData[i])


    # initialize the train, validation, and test data loaders
    trainDataLoader = DataLoader(splittedTrainData, shuffle=True, batch_size=cfg.BATCH_SIZE)
    valDataLoader = DataLoader(valData, batch_size=cfg.BATCH_SIZE)
    testDataLoader = DataLoader(testData, batch_size=cfg.BATCH_SIZE)

    # calculate steps per epoch for training and validation set
    trainSteps = len(trainDataLoader.dataset) // cfg.BATCH_SIZE
    valSteps = len(valDataLoader.dataset) // cfg.BATCH_SIZE
    testSteps = len(testDataLoader.dataset) // cfg.BATCH_SIZE

    print("Number of train data: ", len(trainDataLoader.dataset))
    print("Number of validation data: ", len(valDataLoader.dataset))
    print("Number of Test data: ", len(testDataLoader.dataset))
    print("Train steps: ", trainSteps)
    print("Validation steps: ", valSteps)
    print("Test steps: ", testSteps)

def trainAE():

    # Initialize the model
    print("[INFO] initializing the AE model...")
    model = AE().to(device=device)

    # Initialize our optimizer and loss function
    opt = Adam(model.parameters(), lr=cfg.INIT_LR)

    # Mean Square Error as loss function
    lossFunction = nn.MSELoss()

    # Initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # Measure how long training is going to take
    print("[INFO] training the network...")
    startTime = time.time()
    

    # Loop over our epochs
    for e in range(cfg.EPOCHS):        
        ################################################## Training ####################################################################

        # Set the model in training mode
        model.train()

        # Initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0

        # Initialize the number of correct predictions in the training and validation step
        trainCorrect = 0
        valCorrect = 0

        # Loop over the training set
        for (x, y) in trainDataLoader:    
            
            # Reshaping the image            
            x = x.view(x.size(0), -1)
       
            # Send the input to the device            
            (x, y) = (x.to(device), y.to(device))
            
            # Perform a forward pass
            pred = model(x)
            
            # Calculate the training loss
            pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
            y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
            loss = chamfer_loss(pred, y) * 1000
            
            # Zero out the gradients, perform the backpropagation step, and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Add the loss to the total training loss so far and calculate the number of correct predictions
            totalTrainLoss += loss
            # trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()


        ################################################## Validation ####################################################################

        # Switch off autograd for evaluation
        # Use the torch.no_grad() context to turn off gradient tracking and computation
        # with torch.no_grad():
        #     # Set the model in evaluation mode
        #     model.eval()

        #     # Loop over the validation set
        #     for (x, y) in valDataLoader:
        #         # Reshaping the image            
        #         x = x.view(x.size(0), -1)

        #         # Send the input to the device
        #         (x, y) = (x.to(device), y.to(device))

        #         # Make the predictions
        #         pred = model(x)

        #         # Calculate the validation loss
        #         pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
        #         y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
        #         cdist = chamferDist(pred, y, bidirectional=True)
        #         loss = cdist #+ lossFunction(pred, y)

        #         totalValLoss += loss
        #         # totalValLoss += lossFunction(pred * 10000, y * 10000)

        #         # calculate the number of correct predictions
        #         # valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

        ################################################## Statistics ####################################################################

        # Calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        # avgValLoss = totalValLoss / valSteps

        # Calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(trainDataLoader.dataset)
        # valCorrect = valCorrect / len(valDataLoader.dataset)

        # Update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_acc"].append(trainCorrect)
        # H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        # H["val_acc"].append(valCorrect)

    	# print the model training and validation information
        print(f"[INFO] EPOCH: {e + 1}/{cfg.EPOCHS}")
        print(f"Train loss: {avgTrainLoss:.6f}, Train accuracy: {trainCorrect:.4f}")
        # print(f"Val loss: {avgValLoss:.6f}, Val accuracy: {valCorrect:.4f}\n")



    # finish measuring how long training took
    endTime = time.time()
    print(f"[INFO] total time taken to train the model: {endTime - startTime:.2f}s")


    # Serialize and save the model to disk
    modelPath = cfg.ROOT_DIR + '/Simple_AE.pt'
    torch.save(model, modelPath)
    ################################################### Graphically displaying statistics  ##############################################################

    # Plot the training loss and accuracy
    Utils.save_loss_plot(H)

def trainCAE():
    # Initialize the model
    print("[INFO] initializing the CAE model...")
    model = ().to(device=device)

    # Initialize our optimizer and loss function
    opt = Adam(model.parameters(), lr=cfg.INIT_LR, weight_decay=1e-5)

    # Initialize a dictionary to store training and validation history
    H = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": []
    }

    # Measure how long training is going to take
    print("[INFO] training the network...")
    startTime = time.time()
    

    # Loop over our epochs
    for e in range(cfg.EPOCHS):        
        ################################################## Training ####################################################################

        # Set the model in training mode
        model.train()

        # Initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        totalTestLoss = 0

        # Loop over the training set
        for (x, y) in trainDataLoader:
            # Reshaping the image
            x = Variable(x)

            # Send the input to the device            
            (x, y) = (x.to(device), y.to(device))
            
            # Perform a forward pass
            pred = model(x)
            
            # Calculate the training loss
            pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
            y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
            loss = chamfer_loss(pred, y)
            
            # Zero out the gradients, perform the backpropagation step, and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Add the loss to the total training loss so far and calculate the number of correct predictions
            totalTrainLoss += loss


        ################################################## Validation ####################################################################

        # Switch off autograd for evaluation
        # Use the torch.no_grad() context to turn off gradient tracking and computation
        with torch.no_grad():
            # Set the model in evaluation mode
            model.eval()

            # Loop over the validation set
            for (x, y) in valDataLoader:
                # Reshaping the image
                x = Variable(x)

                # Send the input to the device
                (x, y) = (x.to(device), y.to(device))

                # Make the predictions
                pred = model(x)

                # Calculate the validation loss
                pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
                y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
                loss = chamfer_loss(pred, y)

                totalValLoss += loss


        ################################################## Testing ####################################################################

        # Switch off autograd for evaluation
        # Use the torch.no_grad() context to turn off gradient tracking and computation
        with torch.no_grad():
            # Set the model in evaluation mode
            model.eval()

            # Loop over the validation set
            for (x, y) in testDataLoader:
                # Reshaping the image
                x = Variable(x)

                # Send the input to the device
                (x, y) = (x.to(device), y.to(device))

                # Make the predictions
                pred = model(x)

                # Calculate the validation loss
                pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
                y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
                loss = chamfer_loss(pred, y)

                totalTestLoss += loss



        ################################################## Statistics ####################################################################

        # Calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        avgTestLoss = totalTestLoss / testSteps

        # Update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

    	# print the model training and validation information
        print(f"[INFO] EPOCH: {e + 1}/{cfg.EPOCHS}")
        print(f"Train loss: {avgTrainLoss:.6f}")
        print(f"Val loss: {avgValLoss:.6f}")
        print(f"Test loss: {avgTestLoss:.6f}\n")



    # finish measuring how long training took
    endTime = time.time()
    print(f"[INFO] total time taken to train the model: {endTime - startTime:.2f}s")


    # Serialize and save the model to disk
    modelPath = cfg.ROOT_DIR + '/Convolution_AE.pt'
    torch.save(model, modelPath)


    ################################################### Graphically displaying statistics  ##############################################################
    # Plot the training loss and accuracy
    Utils.save_loss_plot(H)

def trainCVAE():
    # Initialize the model
    print("[INFO] initializing the CVAE model...")
    model = CVAE().to(device=device)

    # Initialize our optimizer and loss function
    opt = Adam(model.parameters(), lr=cfg.INIT_LR)

    # Chamfer distance as loss function
    # lossFunction = nn.BCEWithLogitsLoss()

    # Initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # Measure how long training is going to take
    print("[INFO] training the network...")
    startTime = time.time()
    

    # Loop over our epochs
    for e in range(cfg.EPOCHS):        
        ################################################## Training ####################################################################

        # Set the model in training mode
        model.train()

        # Initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0

        # Initialize the number of correct predictions in the training and validation step
        trainCorrect = 0
        valCorrect = 0

        # Loop over the training set
        for (x, y) in trainDataLoader:
            # Reshaping the image            
            # x = x.view(x.size(0), -1)

            x = Variable(x)

            # Send the input to the device            
            (x, y) = (x.to(device), y.to(device))
            
            # Perform a forward pass
            pred, mu, logvar = model(x)            
            # Calculate the training loss            
            pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
            y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
            cdist = chamfer_loss(pred, y)
            loss = VAELoss(cdist, mu, logvar)
            
            # Zero out the gradients, perform the backpropagation step, and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Add the loss to the total training loss so far and calculate the number of correct predictions
            totalTrainLoss += loss


        ################################################## Validation ####################################################################

        # Switch off autograd for evaluation
        # Use the torch.no_grad() context to turn off gradient tracking and computation
        with torch.no_grad():
            # Set the model in evaluation mode
            model.eval()

            # Loop over the validation set
            for (x, y) in valDataLoader:
                # Reshaping the image
                x = Variable(x)

                # Send the input to the device
                (x, y) = (x.to(device), y.to(device))

                # Perform a forward pass
                pred, mu, logvar = model(x)            
                
                # Calculate the training loss
                pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
                y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
                cdist = chamferDist(pred, y, bidirectional=True)
                loss = VAELoss(cdist, mu, logvar)

                # Add the loss to the total training loss so far and calculate the number of correct predictions
                totalValLoss += loss

        ################################################## Statistics ####################################################################

        # Calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps

        # Calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(trainDataLoader.dataset)
        valCorrect = valCorrect / len(valDataLoader.dataset)

        # Update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_acc"].append(valCorrect)

    	# print the model training and validation information
        print(f"[INFO] EPOCH: {e + 1}/{cfg.EPOCHS}")
        print(f"Train loss: {avgTrainLoss:.6f}, Train accuracy: {trainCorrect:.4f}")
        print(f"Val loss: {avgValLoss:.6f}, Val accuracy: {valCorrect:.4f}\n")



    # finish measuring how long training took
    endTime = time.time()
    print(f"[INFO] total time taken to train the model: {endTime - startTime:.2f}s")


    # Serialize and save the model to disk
    modelPath = cfg.ROOT_DIR + '/CVAE.pt'
    torch.save(model, modelPath)
    ################################################### Graphically displaying statistics  ##############################################################

    # Plot the training loss and accuracy
    Utils.save_loss_plot(H)



def trainCAE_AutoEnd():

    lastLoss = 0.0
    currentLoss = 0.0
    e = 0
    ee = 0
    cnt = 1


    # Initialize the model
    print("[INFO] initializing the CAE model...")
    model = CAE_new().to(device=device)

    # Initialize our optimizer and loss function
    opt = Adam(model.parameters(), lr=cfg.INIT_LR, weight_decay=1e-5)

    # Initialize a dictionary to store training and validation history
    H = {
        "train_loss": [],
        "val_loss": []
    }

    # Measure how long training is going to take
    print("[INFO] training the network...")
    startTime = time.time()
    

    # Loop over our epochs
    # for e in range(cfg.EPOCHS):        
    while True:
        ################################################## Training ####################################################################
                
        lastLoss = currentLoss

        # Set the model in training mode
        model.train()

        # Initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0

        # Loop over the training set
        for (x, y) in trainDataLoader:
            # Reshaping the image
            x = Variable(x)

            # Send the input to the device            
            (x, y) = (x.to(device), y.to(device))
            
            # Perform a forward pass
            pred = model(x)
            
            # Calculate the training loss
            pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
            y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
            loss = chamfer_loss(pred, y)
            
            # Zero out the gradients, perform the backpropagation step, and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Add the loss to the total training loss so far and calculate the number of correct predictions
            totalTrainLoss += loss


        ################################################## Validation ####################################################################

        # Switch off autograd for evaluation
        # Use the torch.no_grad() context to turn off gradient tracking and computation
        with torch.no_grad():
            # Set the model in evaluation mode
            model.eval()

            # Loop over the validation set
            for (x, y) in valDataLoader:
                # Reshaping the image
                x = Variable(x)

                # Send the input to the device
                (x, y) = (x.to(device), y.to(device))

                # Make the predictions
                pred = model(x)

                # Calculate the validation loss
                pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
                y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
                loss = chamfer_loss(pred, y)

                totalValLoss += loss

        ################################################## Statistics ####################################################################

        # Calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps

        # Update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())

    	# print the model training and validation information
        print(f"[INFO] EPOCH: {cnt}/{cfg.EPOCHS}")
        print(f"Train loss: {avgTrainLoss:.6f}")
        print(f"Val loss: {avgValLoss:.6f}")

        currentLoss = avgValLoss
        diffLoss = abs(currentLoss - lastLoss)
        print(f"Difference loss: {diffLoss:.6f}")
        print(f"e: {e:.6f}")
        # print(f"ee: {ee:.6f}\n")
        if (diffLoss <= 0.05):
            e += 1
        # if (currentLoss > lastLoss):
        #     ee += 1

        if e >= 10: # and ee >= 25:
            break
        
        cnt += 1


    # finish measuring how long training took
    endTime = time.time()
    print(f"[INFO] total time taken to train the model: {endTime - startTime:.2f}s")


    # Serialize and save the model to disk
    modelPath = cfg.ROOT_DIR + '/Convolution_AE.pt'
    torch.save(model, modelPath)


    ################################################### Graphically displaying statistics  ##############################################################
    # Plot the training loss and accuracy
    Utils.save_loss_plot(H)



if __name__ == '__main__':
    
    # Train the model
    torch.cuda.empty_cache()
    trainInitialization()
    trainCAE()
