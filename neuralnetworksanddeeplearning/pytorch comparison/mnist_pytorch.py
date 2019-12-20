from __future__ import print_function, division
import torch # Tensor Package (for use on GPU)
from torch.autograd import Variable # for computational graphs
import torch.nn as nn ## Neural Network package
import torch.nn.functional as F # Non-linearities package
import torch.optim as optim # Optimization package
from torch.utils.data import Dataset, TensorDataset, DataLoader # for dealing with data
import torchvision # for dealing with vision data
import torchvision.transforms as transforms # for modifying vision data to run it through models

import matplotlib.pyplot as plt # for plotting
import numpy as np
import time

import dataReader
from feedforward import FeedforwardNet
from laTeX_log import laTeX_log

from skimage import io, transform
from PIL import Image


def load_image(filename):
    img = Image.open(filename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data

def show_image(image, label):
    print('The label is %d' % label)
    plt.imshow(image)
    plt.show()
    

def label_number(labelvector):
    labelvector = [labelvector[i] * i for i in range(len(labelvector))]
    return max(labelvector)

def examples():
    print(np.array(training_data).shape)

    sample = list(training_data)[0]
    sampleArray = sample[0]
    sampleLabel = sample[1]
    print(sample)
    # rearrange data from a simple vector to a matrix representing the image
    # then changing values into standard 8 bit values and converting to image:
    im = Image.fromarray(sampleArray.reshape(28,28) *255)
    show_image(im, label_number(sampleLabel))

def rotation_transformation(array):

    return array

def intensity_transformation(array):

    return array


# set random seed for comparison
RANDOM_SEED = 17
torch.cuda.manual_seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


training_data, validation_data, test_data = dataReader.load_data_wrapper_torch()

training_data, validation_data, test_data = list(training_data), list(validation_data), list(test_data)


# hyperparameters
NUMBER_OF_EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 1e-2
MOMENTUM = 0.8

trainloader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
validationloader = torch.utils.data.DataLoader(validation_data, batch_size = BATCH_SIZE)
testloader = torch.utils.data.DataLoader(test_data, batch_size = BATCH_SIZE)

# network topology and activation functions
sizes_of_layers = [28*28, 50, 30, 20, 10, 10]
activation_functions = [F.relu, F.relu, F.relu, F.relu, F.relu]
activation_functions_string = "[F.relu, F.relu, F.relu, F.relu, F.relu]"

net = FeedforwardNet(sizes_of_layers, activation_functions)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


# initialize the logger to log the training and results
log = laTeX_log( 
    RANDOM_SEED, NUMBER_OF_EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM, 
    sizes_of_layers, activation_functions_string, 
    len(validation_data), len(test_data),
    loss_function, str(optimizer).split(" ")[0] + "()"
)


# Training the neural network
for epoch in range(NUMBER_OF_EPOCHS):
    train_loader_iter = iter(trainloader)
    for batch_idx, (inputs, labels) in enumerate(train_loader_iter):
        inputs, labels = (Variable(inputs).float()), Variable(labels)
        optimizer.zero_grad()
        output = net(inputs)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()
    print("Iteration: " + str(epoch + 1))
    
    correct = 0
    total = 0
    for (inputs, labels) in validationloader:
        outputs = net(Variable(inputs))
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy on the validation set: {} out of {}'.format(correct, total))

    log.add_epoch_result(int(correct))
        

correct = 0
total = 0
for (inputs,labels) in testloader:
    labels = labels
    outputs = net(Variable(inputs))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy on the test set: {} out of {}'.format(correct, total))
log.add_testset_result(int(correct))


log.write_to_file("outputfile_laTeX.txt")




