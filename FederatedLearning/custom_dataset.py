import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

class CustomPathologyDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, train=True, subset_size=30000):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.subset_size = subset_size

        # Select subset of data based on train/test
        if self.train:
            self.data = self.data.sample(n=self.subset_size, random_state=42)
        else:
            self.data = self.data.sample(n=3000, random_state=42)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]  # Assuming the first column is the image file name
        img_path = f"{self.root_dir}/{img_name}.tif"
        image = Image.open(img_path).convert("RGB")  # Convert to RGB if not already in that format
        label = self.data.iloc[idx, 1]  # Assuming the second column is the label

        if self.transform:
            image = self.transform(image)

        return image, label
