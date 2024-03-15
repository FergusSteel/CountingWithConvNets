import  os
import torch
import numpy as np
import pilot_utils
import convert_to_npz
from torch.utils.data import Dataset, DataLoader
import warnings
import matplotlib.image as mpimg
import cv2
from matplotlib.colors import rgb_to_hsv

warnings.filterwarnings("ignore")


class SpreadMNISTDataset(Dataset):
    def __init__(self, num_points, train=True, noise_db=0, path="", std=5.1):
        self.noise_db = noise_db
        self.train = train
        self.dmaps = convert_to_npz.load_data(num_points,  path, train)
        self.path = path

        def noise_function(image, db_iteration, std_dev=std):
    
            new_std = std_dev * (np.sqrt(2)**db_iteration)
            noise = np.abs(np.random.normal(0, new_std, image.shape))
            noisy_image = image + noise
            noisy_image = torch.clamp(noisy_image, 0, 255.)
            return noisy_image
        
        self.noise_function = noise_function
    

    def __len__(self):
        return len(self.dmaps)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.train:
            img_path = os.path.join(os.path.abspath(".."), "data", f"{self.path}train", "x")
        else:
            img_path = os.path.join(os.path.abspath(".."), "data", f"{self.path}test", "x")
        image = cv2.imread(os.path.join(img_path,f"{idx}.png"), 0)
        #image = rgb_to_hsv(image)[:, :, 2]
        dmap = self.dmaps[idx]


        sample = {'image': torch.from_numpy(image), 'dmap': torch.from_numpy(dmap)}

        if self.noise_db != 0:
            sample["image"] = self.noise_function(sample["image"], self.noise_db)
        
        return sample
    
# train_loader = DataLoader(dataset=SpreadMNISTDataset(5), batch_size=1)

# for a in SpreadMNISTDataset(1):
#     print(a["dmap"])