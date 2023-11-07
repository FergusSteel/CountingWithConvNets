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
    print(image.shape)
    return image

def load_batch(n):
    parent_directory = os.path.abspath('..')
    x = np.zeros((n, 256, 256, 1))
    for i in range(n):
        x[i] = load_image(parent_directory + f'/data/train/x/{i}.png')
    y = load_data(n)
    print(y.shape)
    return x, y

load_batch(1)

def conv_model():
    model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 1), padding='same'),
            tf.keras.layers.Conv2D(126, kernel_size=(2, 2), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, (2, 2), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(32, (2, 2), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(10, (1, 1), activation='sigmoid'),
            tf.keras.layers.Reshape((10, 64, 64))
    ]) 
    return model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

model = conv_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
x, y = load_batch(100)
#model.fit(x, y, epochs=5, batch_size=32)
a = model.predict(load_image('../data/train/x/0.png'))
show_density_map(0, a)
