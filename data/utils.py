import numpy as np
import subprocess
import os
import os.path as ospath
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import csv

# Generate #num_points amount of points with min_distance pixels between them
def generate_points(num_points=12, canvas_size=(128,128), min_distance=20):
    points = []

    while len(points) < num_points:
        x = np.random.randint(0, canvas_size[0])
        y = np.random.randint(0, canvas_size[1])

        if all(np.sqrt((x - px) ** 2 + (y - py) ** 2) >= min_distance for px, py in points):
            points.append((x, y))

    return [x[0] for x in points], [y[1] for y in points]

# Load MNIST training shape from keras + normalize
def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    from keras.utils import to_categorical
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

# Utility functions to augment dataset into partitioned ordered sets
from numpy.lib.arraysetops import unique

# Partition ordered dataset
def get_mnist_partitions(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return unique,counts

# Sort and partition dataset.
def augment_dataset(images, labels):
    try:
        labels_sorted = labels[labels.argsort()]
        images_sorted = images[labels.argsort()]
        partitions = get_mnist_partitions(labels_sorted)
        return images_sorted, labels_sorted
    except:
        print("Data Could Not Be Sorted Please Sample Randomly")
        return None
    
def generate_map_config(images, labels, partitions, fname="output", num_digits=24, min_distance=28, prob_density=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], scale_var_prob=0, scale_var_amount=0, rot_var_prob=0, rot_var_deg=0):
    # Generate Coordinates on 256x256 map (leave border at edge)
    xs, ys = generate_points(num_digits, (228,228), min_distance+(images[0].shape[0]*scale_var_amount))
    fig = plt.figure(figsize=(256 / 80, 256 / 80), dpi=80)
    ax = fig.add_axes([0, 0, 1, 1], aspect='equal', frameon=False)
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)

    # Select Sample of images and labels with prob_density
    idxs = []
    vals = np.arange(0,10)
    parts = np.cumsum(partitions)

    lab = 0

    for dig in range(num_digits):
        lab = np.random.choice(vals, 1, p=prob_density)[0]
        # Select random digit with that lab
        max_idx = parts[lab]

        min_idx = 0
        if lab != 0:
            min_idx = parts[lab-1]


        rand_idx = np.random.randint(min_idx, max_idx)
        idxs.append(rand_idx)

    # Place each image on canvas,
    centroid_dict = {}
    for idx, x, y in zip(idxs, xs, ys):
        img = images[idx]
        lab = labels[idx]
        # Check if Scaled/Rotated
        scale = False
        rot = False
        draw = np.random.uniform(0,1)

        if draw <= scale_var_prob:
            scale = True
            up_down = np.random.uniform(-1, 1)
            scale_amount_instance = 1.0 + (up_down * scale_var_amount)
        if draw <= rot_var_prob:
            rot = True
            up_down = np.random.uniform(-1, 1)
            rot_degrees_instance = 0.0 + (up_down * rot_var_deg)
            rotation = Affine2D().rotate_deg(rot_degrees_instance)

        centroid = 0
        if scale:
            centroid = (x + ((img.shape[0]*scale_amount_instance)//2), y + ((img.shape[1]*scale_amount_instance)//2))
            if rot:
                ax.matshow(img, cmap="gray", extent=(x, x + img.shape[0]*scale_amount_instance, y, y + img.shape[1]*scale_amount_instance), transform=rotation)
            else:
                ax.matshow(img, cmap="gray", extent=(x, x + img.shape[0]*scale_amount_instance, y, y + img.shape[1]*scale_amount_instance))
                #ax.plot(x + (img.shape[0]*scale_amount_instance)//2, y + (img.shape[1]*scale_amount_instance)//2, 'ro')
        else:
            centroid = (x + img.shape[0]//2, y + img.shape[1]//2)
            if rot:
                ax.matshow(img, cmap="gray", extent=(x, x + img.shape[0], y, y + img.shape[1]), transform=rotation)
            else:
                ax.matshow(img, cmap="gray", extent=(x, x + img.shape[0], y, y + img.shape[1]))
                #ax.plot(x + (img.shape[0]//2), y + (img.shape[1]//2), 'ro')

        # Construct Dictionary of centroids of objects
        if lab not in centroid_dict.keys():
            centroid_dict[lab] = [centroid]
        else:
            centroid_dict[lab].append(centroid)

    # for centroid in centroid_dict.values():
    #   for centre in centroid:
    #     ax.plot(centre[0], centre[1], 'ro')

    # Save image
    fig.set_facecolor('black')
    fig.savefig(f'train/x/{fname}.png', transparent=False)
    plt.close()

    # Save mapping
    with open(f"train/y/{fname}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(10):
            # Flatten the coordinates list and convert them to strings
            if i in centroid_dict.keys():
                coordinates = [f"{int(x)}|{int(y)}" for x, y in centroid_dict[i]]
            else:
                coordinates = []
            # Write the key and coordinates as a single row
            row = [i] + coordinates
            writer.writerow(row)