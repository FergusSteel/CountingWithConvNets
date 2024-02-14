from utils import *
import argparse
from tqdm import tqdm
import os

# ARG PARSING
def pars_cfg():
    parser = argparse.ArgumentParser(
        prog="dataset_generator",
        description = "Generation Config for Multi-MNIST dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train", type=int, default=1, help="Use training distribution (1) or test (0)")
    parser.add_argument("--file_name", type=str, default="", help="Name of the file to save.")
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
    file_path=config.file_name

    # setupfile strucutre
    target_file_path_root = f"{file_path}{'train' if config.train == 1 else 'test'}"
    print(f"target_file_path_root: {target_file_path_root}")

    if not os.path.exists(target_file_path_root):
        os.makedirs(target_file_path_root)
    if not os.path.exists(f"{target_file_path_root}/x"):
        os.makedirs(f"{target_file_path_root}/x")
    if not os.path.exists(f"{target_file_path_root}/y"):
        os.makedirs(f"{target_file_path_root}/y")
                
    #normalise the probas
    pdtot = sum(prob_density)
    for i in range(len(prob_density)):
        prob_density[i] = prob_density[i] / pdtot

    for i in tqdm(range(0, num_images)):
        n = np.random.randint(min_n, max_n + 1)
        generate_map_config(images, labels, partitions, i, n, min_distance, prob_density, scale_var_prob, scale_var_amount, train=config.train, arg=file_path)


if __name__ == "__main__":
    config = pars_cfg()
    assert config.min_distance > 14 and config.min_distance < 40
    assert config.max_n < 32
    
    generator(config)
