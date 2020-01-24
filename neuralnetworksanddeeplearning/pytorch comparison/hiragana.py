from __future__ import print_function, division
import torch # Tensor Package (for use on GPU)
from torch.autograd import Variable # for computational graphs
import torch.nn as nn ## Neural Network package
import torch.nn.functional as F # Non-linearities package
import torch.optim as optim # Optimization package
from torch.utils.data import Dataset, TensorDataset, DataLoader # for dealing with data
import torchvision # for dealing with vision data
import torchvision.transforms as transforms # for modifying vision data to run it through models


import numpy as np
import time

import dataReader
from feedforward import FeedforwardNet
from laTeX_log import laTeX_log
from training import train_network
from MNIST_Dataset import MNIST_Dataset
import sampling


def test_network():
    net.eval()
    correct = 0
    total = 0
    for (inputs,labels) in testloader:
        if CUDA_FLAG:
            labels = labels.cuda()
            outputs= net(Variable(inputs.cuda()))
        else:
            outputs = net(Variable(inputs))
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return (correct, total)

CUDA_FLAG = False # determines whether the GPU is used for calculations
TRAINING_ERROR = True # determines whether or not the accuracy on the trainingset will be calculated and logged
LOAD_STATE = False # determines whether or not the initial state of the NN's weights and biases will be loaded from a file

# set random seed for comparison
RANDOM_SEED = 392181423
torch.cuda.manual_seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)

# hyperparameters
NUMBER_OF_EPOCHS = 15
BATCH_SIZE = 50
LEARNING_RATE = 2 * 1e-1
MOMENTUM = 0.5
USE_DROPOUT = False
dropout_probability = 0.5

# network topology and activation functions
sizes_of_layers = [784, 150, 50, 30, 10]
activation_functions = [ F.relu, F.relu, F.relu, F.relu]
activation_functions_string = "[F.relu, F.relu, F.relu, F.relu]"

# Load and prepare data for use
training_data, validation_data, test_data = dataReader.load_hiragana()





training_dataset = MNIST_Dataset(training_data)



trainloader = torch.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
validationloader = torch.utils.data.DataLoader(validation_data, batch_size = BATCH_SIZE)
testloader = torch.utils.data.DataLoader(test_data, batch_size = BATCH_SIZE)

# creating the neural network
net = FeedforwardNet(sizes_of_layers, activation_functions, USE_DROPOUT=USE_DROPOUT, dropout_probability=dropout_probability)
if(CUDA_FLAG):
    net = net.cuda()

for data in training_data[8:20]:
    sampling.examples(data)




