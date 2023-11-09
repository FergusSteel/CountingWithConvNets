import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from convert_to_npz import load_data
from convert_to_npz import show_density_map

# Load data
def load_image(image_path):
    parent_directory = os.path.abspath('..')
    image = load_img(image_path, color_mode='grayscale', target_size=(256, 256))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0
    return image

def load_batch(n):
    parent_directory = os.path.abspath('..')
    x = np.zeros((n, 256, 256, 1))
    for i in range(n):
        x[i] = load_image(parent_directory + f'/data/train/x/{i}.png')
    y = load_data(n)
    return x, y

def conv_model():
    model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 1), padding='same'),
            tf.keras.layers.Conv2D(126, kernel_size=(3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(10, (1, 1), activation='sigmoid'),
            tf.keras.layers.Reshape((10,64,64))
    ]) 
    return model

# def loss_function(y_true, y_predict):
#     for map in batch

# model = conv_model()
# print(model.summary())
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
# x, y = load_batch(400)
# model.fit(x, y, epochs=1, batch_size=20)
# a = model.predict(load_image('../data/train/x/0.png'))

# print(a[0].shape)
