# %%
import os

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import *


class MIT67Dataset(Dataset):

    def __init__(self,
                 data_root='/',
                 limited=True,
                 split='train',
                 transforms=None) -> None:
        super(MIT67Dataset, self).__init__()
        self.data_root = data_root
        self.limited = limited
        self.transform = transforms
        self.label2id = {
            cls_name: idx
            for idx, cls_name in enumerate(
                USED_ROOM_TYPES if limited else ROOM_TYPES)
        }

        self.split = split
        list_file = f"{self.data_root}/TrainImages.txt" if split == 'train' else f"{self.data_root}/TestImages.txt"
        with open(list_file) as f:
            raw_list = [i.strip().split('/') for i in f.readlines()]
        self.data_list = [(f"{self.data_root}/Images/{item[0]}/{item[1]}",
                           self.label2id[item[0]]) for item in raw_list
                          if not limited or item[0] in USED_ROOM_TYPES]

    def __getitem__(self, index):
        image = Image.open(self.data_list[index][0]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = torch.tensor(self.data_list[index][1])
        return (image, target, self.data_list[index])

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    dataset = MIT67Dataset('/home/tb5zhh/MIT67', split='test')
    print(len(dataset))
# %%
