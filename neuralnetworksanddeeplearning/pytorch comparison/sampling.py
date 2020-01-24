import numpy as np
from skimage import io, transform
from PIL import Image
import matplotlib.pyplot as plt # for plotting
import torchvision.transforms as transforms # for modifying vision data to run it through models

sample_translation = transforms.RandomAffine(0, translate=(0.15, 0.15))
sample_rotate = transforms.RandomRotation(10)
sample_scale = transforms.RandomAffine(0, scale=(0.7, 1.3))

def load_image(filename):
    img = Image.open(filename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data

def show_image(image, label):
    print('The label is %d' % label)
    plt.imshow(image)
    plt.show()
    

def label_number(label):
    labelvector = [1 if(i == label) else 0 for i in range(label)]
    return max(labelvector)

def examples(training_data):
    print(np.array(training_data).shape)

    sample = list(training_data)[0]
    sampleArray = sample[0]
    sampleLabel = sample[1]
    print(sample)
    # rearrange data from a simple vector to a matrix representing the image
    # then changing values into standard 8 bit values and converting to image:
    im = Image.fromarray(sampleArray.reshape(28,28) *255)
    show_sample_transformations(im, sampleLabel)

def show_sample_transformations(image, label):
    show_image(image, label)
    image_translated = sample_translation(image)
    show_image(image_translated, label)
    image_rotated = sample_rotate(image)
    show_image(image_rotated, label)
    image_scaled = sample_scale(image)
    show_image(image_scaled, label)
    
