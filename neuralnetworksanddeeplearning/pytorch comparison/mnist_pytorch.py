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







def adjust_learning_rate(optimizer, epoch, lr_decay, lr_decay_epoch):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        return optimizer
    
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer
    




transformation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(degrees = 6.3, translate = (0.05, 0.05)),
    transforms.ToTensor()])

TRAINING_ERROR = True


# set random seed for comparison
RANDOM_SEED = 22
torch.cuda.manual_seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)

# hyperparameters
NUMBER_OF_EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 1 * 1e-2
MOMENTUM = 0.8


# Load and prepare data for use
training_data, validation_data, test_data = dataReader.load_data_wrapper_torch()
training_data, validation_data, test_data = list(training_data), list(validation_data), list(test_data)
training_dataset = MNIST_Dataset(training_data, transform= transformation)

trainloader = torch.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
validationloader = torch.utils.data.DataLoader(validation_data, batch_size = BATCH_SIZE)
testloader = torch.utils.data.DataLoader(test_data, batch_size = BATCH_SIZE)

# network topology and activation functions
sizes_of_layers = [28*28, 75, 50, 30, 10]
activation_functions = [F.relu, F.relu, F.relu, F.relu]
activation_functions_string = "[F.relu, F.relu, F.relu, F.relu]"

net = FeedforwardNet(sizes_of_layers, activation_functions)


loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


# initialize the logger to log the training and results
log = laTeX_log( 
    RANDOM_SEED, NUMBER_OF_EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM, 
    sizes_of_layers, activation_functions_string, 
    len(training_data), len(validation_data), len(test_data),
    loss_function, str(optimizer).split(" ")[0] + "()", LOG_TRAINING_SET=TRAINING_ERROR
)


best_validation_rate = 0
# Training the neural network
for epoch in range(NUMBER_OF_EPOCHS):
    train_loader_iter = iter(trainloader)
    for batch_idx, (inputs, labels) in enumerate(train_loader_iter):
        inputs, labels = Variable(inputs).float(), Variable(labels)
        optimizer.zero_grad()
        output = net(inputs)
        loss = loss_function(output, labels)
        loss.backward()
        # print(net.module_list[0].weight.grad)
        optimizer.step()
    print("Iteration: " + str(epoch + 1))
    
    if(TRAINING_ERROR):
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            outputs = net(Variable(inputs))
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted==labels).sum()
        print("Accuracy on training set: {} out of {}".format(correct,total))
        log.add_trainingset_result(int(correct))

    correct = 0
    total = 0
    for (inputs, labels) in validationloader:
        outputs = net(Variable(inputs))
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy on the validation set: {} out of {}'.format(correct, total))

    
    # multiply learning rate with 0.1 every 10 epochs
    optimizer = adjust_learning_rate(optimizer, epoch, 0.1, 10)

    log.add_validationset_result(int(correct))

    if(correct/total >= best_validation_rate):
        net.save_NN_state("NN_States/best_state.pt")

        

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




