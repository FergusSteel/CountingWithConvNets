import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# Define a function to create a 2D Gaussian kernel
def create_density_gaussian(points):
    map  = np.zeros((10,64,64))
    for i in range(10):
        coords = points[i]
        for coord in coords:
            map[i][int(coord[1])//4][int(coord[0])//4] = 1
    
        map[i] = np.flipud(gaussian_filter(map[i], sigma=2))
    
    return map
    

# Read each CSV file and create density maps
def read_csv(n):
    filename = f"train/y/{n}.csv"
    parent_directory = os.path.abspath('..')  # Get the absolute path of the parent directory
    file_path = os.path.join(parent_directory, 'data', 'train', 'y', f"{n}.csv")  # Specify the relative path to your file

    with open(file_path, "r", newline="") as f:
        csv_reader = csv.reader(f)
        i = 0
        points = []
        for row in csv_reader:
            points.append([])
            skipval = 0
            for value in row:
                if skipval == 0:
                    skipval = 1
                    continue
                points[i].append(value.split("|"))
            i += 1

    return points
    

# Show density map ontop of image
def show_density_map(n, map):
    parent_directory = os.path.abspath('..')  # Get the absolute path of the parent directory
    file_path = os.path.join(parent_directory, 'data', 'train')
    img = mpimg.imread(os.path.join(file_path, "x", f"{n}.png"))
    plt.imshow(img, alpha=1)
    plt.imshow(map, cmap='gray', alpha=0.5, extent=[0, 256, 256, 0])
    plt.show()


def load_data(num_images):
    y = np.zeros((num_images, 10, 64, 64))
    print("Loading data...")
    print("This may take a while...")
    for i in tqdm(range(num_images)):
        points = read_csv(i)
        density_map = create_density_gaussian(points)
        y[i] = density_map

    return y        