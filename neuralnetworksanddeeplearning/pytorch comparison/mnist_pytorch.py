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

def downscale_array_by_half(array, mode="rgb"):
    # list of two adjacent column elements:
    
    column_array = np.array([list(zip(array[j], array[j+1])) if j+1 < array.shape[0] else list(zip(array[j], array[j]))
                            for j in range(0, array.shape[0], 2)])
    # list of the two adjacest row elements in the column_list:
    four_tuple_list = [(column_list[i][j][0], column_list[i][j][1], column_list[i][j+1][0], column_list[i][j+1][1]) 
                        if j+1 < column_list.shape[1] 
                        else (column_list[i][j][0], column_list[i][j][1], column_list[i][j][0], column_list[i][j][1]) 
                        for i in range(0, column_list.shape[0], 1) for j in range(0, column_list.shape[1], 2)]
    four_tuple_array = np.array(four_tuple_list).reshape(int(array.shape[0]/2 + 0.5), int(array.shape[1]/2 + 0.5), 4)
    down_scaled_array = np.zeros([four_tuple_list.shape[0], four_tuple_list.shape[1]])
    if (mode=="bw"):
    for i in range(four_tuple_list.shape[0]):
        down_scaled_array[i] = [(list(x)[0] + list(x)[1] + list(x)[2] + list(x)[3] )/4 for x in list(four_tuple_list[i])]

    if (mode=="rgb"):
    
    return down_scaled_array

    


akane_array = load_image("testimages/Akane.jpg")
print(akane_array[20][0:20])

for i in range(column_list):
    print(list(list(column_list)[i]))





