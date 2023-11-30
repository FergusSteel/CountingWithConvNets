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
    def __init__(self, num_points, train=True, transform=None):
        self.transform = transform
        self.dmaps = convert_to_npz.load_data(num_points)
        
    

    def __len__(self):
        return len(self.dmaps)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.train:
            img_path = os.path.join(os.path.abspath(".."), "data", "train", "x")
        else:
            img_path = os.path.join(os.path.abspath(".."), "data", "test", "x")
        image = cv2.imread(os.path.join(img_path,f"{idx}.png"), 0)
        #image = rgb_to_hsv(image)[:, :, 2]
        dmap = self.dmaps[idx]

        sample = {'image': torch.from_numpy(image), 'dmap': torch.from_numpy(dmap)}

        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
# train_loader = DataLoader(dataset=SpreadMNISTDataset(5), batch_size=1)

# for a in SpreadMNISTDataset(1):
#     print(a["dmap"])