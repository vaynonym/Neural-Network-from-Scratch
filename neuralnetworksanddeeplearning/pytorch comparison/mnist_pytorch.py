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

import os
import pandas as pd
from skimage import io, transform
from PIL import Image
import imageio

def load_image(filename):
    img = Image.open(filename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data

def show_image(image, label):
    # print('The label is {%2}')
    plt.imshow(image)
    plt.show()
    time.sleep(5)


training_data, validation_data, test_data = dataReader.load_data_wrapper()

sample = list(training_data)[0]
sampleArray = sample[0]
sampleLabel = sample[1]
'''print("image_data:")
print(sampleArray)
print("label:")
print(sampleLabel)
'''
# rearrange data from a simple vector to a matrix representing the image
# then changing values into standard 8 bit values and converting to image:
im = Image.fromarray(sampleArray.reshape(28,28) *255)
show_image(im)

















