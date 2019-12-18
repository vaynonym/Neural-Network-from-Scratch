import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader # for dealing with data
import torchvision # for dealing with vision data
import torchvision.transforms as transforms # for modifying vision data to run it through models

import matplotlib.pyplot as plt # for plotting
import numpy as np

x = torch.Tensor([[0,0,1,1],
                 [0,1,1,0],
                 [1,0,1,0],
                 [1,1,1,1]])

target_y = torch.Tensor([0,1,1,0])
# now, instead of having 1 data sample, we have 4 (oh yea, now we're in the big leagues)
# but, pytorch has a DataLoader class to help us scale up, so let's use that.

inputs = x
labels = target_y

train = TensorDataset(inputs, labels) # here we're just 
# putting our data samples into a tiny Tensor dataset

trainloader = DataLoader(train, batch_size = 2, shuffle=False)

linear_layer1 = nn.Linear(4, 1)

EPOCHS = 10
EPSILON = 1e-1
loss_function = nn.MSELoss()
optimizer = optim.SGD(linear_layer1.parameters(), lr = EPSILON)

for epoch in range(EPOCHS):
    # here's the iterator we use to iterate over our training set
    train_loader_iter = iter(trainloader)

    # here we split apart our data so we can run it
    for batch_idx, (inputs, labels) in enumerate(train_loader_iter):
        linear_layer1.zero_grad()
        inputs, labels = Variable(inputs.float()), Variable(labels.float())
        predicted_y = linear_layer1(inputs)
        loss = loss_function(predicted_y, labels)
        loss.backward()
        optimizer.step()
        print("----------------------------------------")
        print("Output (UPDATE: Epoch #)" + str(epoch + 1) + ", Batch#" + 
            str(batch_idx + 1) + "):")
        print(linear_layer1(Variable(x)))
        print("Should be getting closer to [0,1,1,0]...")
        # but some of them aren't! we need a model that fits better...
        # next up, we'll convert this model from regression to a NN

print("------------------------------------------")     