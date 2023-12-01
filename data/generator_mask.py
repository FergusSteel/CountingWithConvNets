from utils_mask import *
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import os

# ARG PARSING
def pars_cfg():
    parser = argparse.ArgumentParser(
        prog="dataset_generator",
        description = "Generation Config for Multi-MNIST dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train", type=int, default=1, help="Use training distribution (1) or test (0)")
    parser.add_argument("--num_images", type=int, default=10000, required=True, help="Number of images to generate.")
    parser.add_argument("--min_n", type=int, default=0, help="Max number of digits in each image.")
    parser.add_argument("--max_n", type=int, default=16, help="Min number of digits in each image.")
    parser.add_argument("--min_distance", type=int, default=28, help="Minimum distance between each digit in an image.")
    parser.add_argument("--prob_density", type=float, nargs=10, default=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], help="Probability mass function, ten arguments, sums to 1.0")
    parser.add_argument("--scale_var_prob", type=float, default=0.0, help="Chance that each digit has to scale. [0.0, 1.0]")
    parser.add_argument("--scale_var_amount", type=float, default=0.0, help="Maximum that each digit has to scale.")


    cfg = parser.parse_args()
    return cfg

# Config will contain the parameters to alter the max digits, variance, density, scale and probability of the images - pulled from argparser most likely
def generator(config):
    train,test = load_mnist()

    if config.train == 1:
        images, labels = train
    else:
        images, labels = test
    
    labels = np.argmax(labels, axis=1)[::]
    images, labels = augment_dataset(images, labels)
    partitions = get_mnist_partitions(labels)[1]


    # Fetch from config
    num_images=config.num_images
    min_n = config.min_n
    max_n = config.max_n
    min_distance=config.min_distance
    prob_density=config.prob_density
    scale_var_prob=config.scale_var_prob
    scale_var_amount=config.scale_var_amount

    for i in tqdm(range(0, num_images)):
        # Select the amount of digits in an image
        n = np.random.randint(min_n, max_n + 1)
        generate_map_config(images, labels, partitions, i, n, min_distance, prob_density, scale_var_prob, scale_var_amount)


if __name__ == "__main__":
    config = pars_cfg()
    assert config.min_distance > 14 and config.min_distance < 40
    assert config.max_n < 32
    assert np.sum(config.prob_density) == 1.0
    
    generator(config)


    # Test drawing:
    def read_csv(n):
        file_path = os.path.join("mask", f"{n}.csv")  # Specify the relative path to your file

        with open(file_path, "r", newline="") as f:
            csv_reader = csv.reader(f)
            i = 0
            masks = []
            for row in csv_reader:
                masks.append([])
                skipval = 0
                for value in row:
                    if skipval == 0:
                        skipval = 1
                        continue
                    masks[i].append(value)
                    print(type(value))
                masks[i] = np.array(masks[i])
                i += 1

        return masks
    

    # Show density map ontop of image
    def show_density_map(img, dmap):
        # parent_directory = os.path.abspath('..')  # Get the absolute path of the parent directory
        # file_path = os.path.join(parent_directory, 'data', 'train')
        # img = mpimg.imread(os.path.join(file_path, "x", f"{n}.png"))
        #print(dmap)
        #print(dmap)
        plt.imshow(img, alpha=1, cmap="gray")
        plt.imshow((dmap), cmap="plasma", alpha=1, extent=[0, 256, 256, 0])
        plt.show()


    #dmap = read_csv(0)
    show_density_map(plt.imread("mask/0.png"), np.load("mask/0.npy", allow_pickle=True)[0][0])