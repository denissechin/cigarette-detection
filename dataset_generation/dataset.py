import os
from typing import Union

import cv2
import torch
from torch.utils.data import Dataset


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, image_path: str, transforms=None):
        self.images = [os.path.join(image_path, filepath) for filepath in sorted(os.listdir(image_path))]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: Union[int, dict]) -> dict:
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
        return {
            'image': image,
        }

