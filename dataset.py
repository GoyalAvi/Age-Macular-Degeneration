# dataset.py
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class OCTLineDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = [
            f for f in os.listdir(image_dir) if f.endswith(".tif")
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(
            self.mask_dir, image_name.replace(".tif", "_mask.npy")
        )

        # Load and normalize image
        image = Image.open(image_path).convert("L")
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.tensor(image).unsqueeze(0)  # shape: [1, H, W]

        # Load corresponding mask (already labeled with 0, 1, 2)
        mask = np.load(mask_path)  # shape: [H, W]
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask
