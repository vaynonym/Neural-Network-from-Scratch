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
from MNIST_Dataset import MNIST_Dataset
from training import train_network

        
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

 
transformation = transforms.Compose([
    transforms.ToTensor(),
    # add noise which adds or subtracts proportionally to x. This way, 
    # every value that started as 0 will remain 0 and we won't have negative values 
    transforms.Lambda(lambda x : x + 0.03 * x * (2 * (torch.rand(x.shape)) - 1)),
    transforms.ToPILImage(),
    transforms.RandomAffine(degrees = 5.0, translate = (0.04, 0.04), scale=(0.9, 1.1)),
    transforms.ToTensor()
    ])


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
training_data, validation_data, test_data = dataReader.load_data_wrapper_torch()
training_data, validation_data, test_data = list(training_data), list(validation_data), list(test_data)
training_dataset = MNIST_Dataset(training_data, transform=transformation)

trainloader = torch.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
validationloader = torch.utils.data.DataLoader(validation_data, batch_size = BATCH_SIZE)
testloader = torch.utils.data.DataLoader(test_data, batch_size = BATCH_SIZE)

# creating the neural network
net = FeedforwardNet(sizes_of_layers, activation_functions, USE_DROPOUT=USE_DROPOUT, dropout_probability=dropout_probability)
if(CUDA_FLAG):
    net = net.cuda()


loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# Determine where the state of the neural network will be saved or loaded
SAVE_STATE_PATH = "NN_States/" + str(sizes_of_layers) + "_best_state.pt"
LOAD_STATE_PATH = "NN_States/" + "" + "best_state.pt"
if(LOAD_STATE):
    # loads all layers
    # net.load_NN_state(LOAD_STATE_PATH)
    # loads all layers excluding the last one and works even if model has more layers afterward
    net.load_partial_NN_state(LOAD_STATE_PATH)

# initialize the logger to log the training and results
log = laTeX_log( 
    RANDOM_SEED, NUMBER_OF_EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM, 
    sizes_of_layers, activation_functions_string, 
    len(training_data), len(validation_data), len(test_data),
    loss_function, str(optimizer).split(" ")[0] + "()", LOG_TRAINING_SET=TRAINING_ERROR
)

# the big function
train_network(net, optimizer, NUMBER_OF_EPOCHS, loss_function, trainloader, validationloader, log, SAVE_STATE_PATH, CUDA_FLAG, TRAINING_ERROR)

# testing state of the network at the end of training
correct, total = test_network()
print('Accuracy on the test set after training: {} out of {}'.format(correct, total))

# testing state of the network at the time of best performance on the validation set
net.load_NN_state(SAVE_STATE_PATH)
correct, total = test_network()
print('Best Iteration: Accuracy on the test set: {} out of {}'.format(correct, total))
log.add_testset_result(int(correct))

log.write_to_file("outputfile_laTeX.txt")




