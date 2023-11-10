import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import torch
import torch.nn as nn  
from torch.nn.functional import relu
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from convert_to_npz import load_data
from convert_to_npz import show_density_map, create_density_gaussian, read_csv

# Load data
def load_image(image_path):
    parent_directory = os.path.abspath('..')
    image = load_img(image_path, color_mode='grayscale', target_size=(256, 256))
    image = img_to_array(image)
    image /= 255.0
    image = image.reshape(1, 256, 256)

    return image

def load_batch(n):
    parent_directory = os.path.abspath('..')
    x = np.zeros((n, 1, 256, 256))
    for i in range(n):
        x[i] = load_image(parent_directory + f'/data/train/x/{i}.png')
    y = load_data(n)
    return x, y

def conv_model():
    model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(256, kernel_size=(9, 9), activation='relu', input_shape=(256, 256, 1), padding='same'),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            #tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(10, (1, 1), activation='sigmoid'),
            tf.keras.layers.Reshape((10,64,64))
    ]) 
    return model

class TorchUNetModel(nn.Module): # S/O "Cook your first Unet in PyTorch" - 
    def __init__(self, classes=10):
        super().__init__()

        #Encoding module 5x(Conv->Conv->MaxPool)
        # Layer 1
        self.layer1conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1) # output: 254x254x62
        self.layer1conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 250x250x62
        self.layer1pool = nn.MaxPool2d(kernel_size=2, stride=2) # output: 125x125x62

        # Layer 2
        self.layer2conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 123x123x128
        self.layer2conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 121x121x128
        self.layer2pool = nn.MaxPool2d(kernel_size=2, stride=2) # output: 60x60x128

        # Layer 3
        self.layer3conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 58x58x256
        self.layer3conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 56x56x256
        self.layer3pool = nn.MaxPool2d(kernel_size=2, stride=2) # output: 28x28x256

        # Layer 4
        print("Layer 4")
        self.layer4conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 26x26x512
        self.layer4conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 24x24x512
        self.layer4pool = nn.MaxPool2d(kernel_size=2, stride=2) # output: 12x12x512

        #layer 5
        print("Layer 5")
        self.layerbotconv1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 10x10x1024
        self.layerbotconv2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 8x8x1024

        # Decoding Moduel 4x(UpConv->Conv->Conv)
        # Layer 1
        print("Up Conv oclock")
        self.layer5upconv = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # output: 24x24x512
        self.layer5conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.layer5conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Layer 2
        self.layer6upconv = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.layer6conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.layer6conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Layer 3
        self.layer7upconv = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.layer7conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.layer7conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Layer 4
        self.layer8upconv = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.layer8conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.layer8conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output Layer
        self.output = nn.Conv2d(64, classes, kernel_size=1)

    def forward(self, x):
        # Encoding Module
        # Layer 1
        x11 = relu(self.layer1conv1(x))
        x12 = relu(self.layer1conv2(x11))
        xp1 = self.layer1pool(x12)

        # Layer 2
        x21 = relu(self.layer2conv1(xp1))
        x22 = relu(self.layer2conv2(x21))
        xp2 = self.layer2pool(x22)

        # Layer 3
        x31 = relu(self.layer3conv1(xp2))
        x32 = relu(self.layer3conv2(x31))
        xp3 = self.layer3pool(x32)

        # Layer 4
        x41 = relu(self.layer4conv1(xp3))
        x42 = relu(self.layer4conv2(x41))
        xp4 = self.layer4pool(x42)

        # Layer 5
        x51 = relu(self.layerbotconv1(xp4))
        x52 = relu(self.layerbotconv2(x51))

        # Decoding Module
        # Layer 6
        xu6 = self.layer5upconv(x52)
        xuc6 = torch.cat([xu6, x42], dim=1)
        x61 = relu(self.layer5conv1(xuc6))
        x62 = relu(self.layer5conv2(x61))

        # Layer 7
        xu7 = self.layer6upconv(x62)
        xuc7 = torch.cat([xu7,  x32], dim=1)
        x71 = relu(self.layer6conv1(xuc7))
        x72 = relu(self.layer6conv2(x71))

        # Layer 8
        xu8 = self.layer7upconv(x72)
        xuc8 = torch.cat([xu8, x22], dim=1)
        x81 = relu(self.layer7conv1(xuc8))
        x82 = relu(self.layer7conv2(x81))

        # Layer 9
        xu9 = self.layer8upconv(x82)
        xuc9 = torch.cat([xu9, x12], dim=1)
        x91 = relu(self.layer8conv1(xuc9))
        x92 = relu(self.layer8conv2(x91))

        # Output Layer
        x_out = self.output(x92)

        return x_out
    


    
# model = conv_model()
# print(model.summary())
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
# x, y = load_batch(1)
# model.fit(x, y, epochs=20, batch_size=1)
# a = model.predict(load_image('../data/train/x/0.png'))

# show_density_map(x[0], a[0])

# x, y = load_batch(5)

# for i in range(5):
#     show_density_map(x[i], y[i])

# print(a[0].shape)

# show_density_map(load_image("../data/train/x/0.png")[0], create_density_gaussian(read_csv(0)))

# print(sum(sum(sum(y[0]))))
