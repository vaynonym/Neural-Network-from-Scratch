import numpy as np
from skimage import io, transform
from PIL import Image
import matplotlib.pyplot as plt # for plotting

def load_image(filename):
    img = Image.open(filename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data

def show_image(image, label):
    print('The label is %d' % label)
    plt.imshow(image, cmap="gray")
    plt.show()
    

def label_number(labelvector):
    labelvector = [labelvector[i] * i for i in range(len(labelvector))]
    return max(labelvector)

def examples(sample):
    sampleArray = sample[0]
    sampleLabel = sample[1]
    print("Label:")
    print(sampleLabel)
    print("array")
    print(sampleArray)
    # rearrange data from a simple vector to a matrix representing the image
    # then changing values into standard 8 bit values and converting to image:
    im = Image.fromarray(sampleArray.reshape(28,28), mode="L")
    show_image(im, sampleLabel)