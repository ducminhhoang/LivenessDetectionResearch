import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def __len__(self):
        """Trả về số lượng mẫu trong dataset."""
        raise NotImplementedError("You need to implement this method for each dataset")

    def __getitem__(self, idx):
        """
        Tải và trả về một mẫu từ dataset.
        Args:
            idx (int): Chỉ số của mẫu cần tải.

        Returns:
            tuple: (image, label) tương ứng với ảnh và nhãn của nó.
        """
        raise NotImplementedError("You need to implement this method for each dataset")

    def train_dataloader(self):
        raise NotImplementedError("You need to implement this method for each dataset")

    def valid_dataloader(self):
        raise NotImplementedError("You need to implement this method for each dataset")