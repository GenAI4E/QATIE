from PIL import Image
import numpy as np
from glob import glob
from typing import List, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import lightning as pl

class DPEDData(Dataset):
    def __init__(self, data_dir: Union[List[Tuple[str, str]], Tuple[str, str]], split: str='train', patch_data: bool = False):
        """
        data_dir can be either a list of tuples (phone, path) or a single tuple (phone, path)
        e.g., data_dir = [("iphone", "/path/to/iphone"), ("canon", "/path/to/canon")]

        split can be either 'train' or 'test'

        patch_data:
          - True: use ``test_data/patches/<phone>`` / ``test_data/patches/canon`` (test split).
          - False: use full-resolution pairs under ``<path>/<phone>/*.jpg`` and ``<path>/canon/*.jpg``.
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.patch_data = patch_data
        self.image_paths = []
        self.canon_image_paths = []
        if patch_data:
            if isinstance(data_dir, list):
                self.image_paths = []
                for phone, path in data_dir:
                    if split == 'train':
                        self.image_paths += glob(str(path / "training_data" / phone / "*.jpg"))
                        self.canon_image_paths += glob(str(path / "training_data" / "canon" / "*.jpg"))
                    else:
                        self.image_paths += glob(str(path / "test_data" / "patches" / phone / "*.jpg"))
                        self.canon_image_paths += glob(str(path / "test_data" / "patches" / "canon" / "*.jpg"))
            elif isinstance(data_dir, tuple):
                if split == 'train':
                    self.image_paths = glob(str(data_dir[1] / "training_data" /  data_dir[0] / "*.jpg"))
                    self.canon_image_paths = glob(str(data_dir[1] / "training_data" / "canon" / "*.jpg"))
                else:
                    self.image_paths = glob(str(data_dir[1] / "test_data" / "patches" / data_dir[0] / "*.jpg"))
                    self.canon_image_paths = glob(str(data_dir[1] / "test_data" / "patches" / "canon" / "*.jpg"))
            else:
                raise ValueError("data_dir should be either a list of tuples or a single tuple.")
        else:
            if isinstance(data_dir, list):
                self.image_paths = []
                for phone, path in data_dir:
                    self.image_paths += glob(str(path / phone / "*.jpg"))
                    self.canon_image_paths += glob(str(path / "canon" / "*.jpg"))
            elif isinstance(data_dir, tuple):
                self.image_paths = glob(str(data_dir[1] / data_dir[0] / "*.jpg"))
                self.canon_image_paths = glob(str(data_dir[1] / "canon" / "*.jpg"))
            else:
                raise ValueError("data_dir should be either a list of tuples or a single tuple.")

        self.train_augs = T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])      
        self.val_augs = T.Compose([
            T.ToTensor(),
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = np.array(image).astype(np.float32)
    
        canon_image = Image.open(self.canon_image_paths[idx])
        canon_image = np.array(canon_image).astype(np.float32)
    
        if self.split == 'train':
            # Stack, apply same random transform, then split
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.train_augs(image) / 255.0
            torch.manual_seed(seed)
            canon_image = self.train_augs(canon_image) / 255.0
        else:
            image = self.val_augs(image) / 255.0
            canon_image = self.val_augs(canon_image) / 255.0
    
        return image, canon_image

class DPEDDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4, patch_data: bool = False):
        """``patch_data`` must match evaluation mode: True for patch test set, False for full-HD images."""
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_data = patch_data
    def setup(self, stage=None):
        print("stage = ", stage)
        if stage == 'fit' or stage is None:
            self.train_dataset = DPEDData(self.data_dir, split='train', patch_data=self.patch_data)
            self.val_dataset = DPEDData(self.data_dir, split='test', patch_data=self.patch_data)
        if stage == 'test' or stage is None:
            self.test_dataset = DPEDData(self.data_dir, split='test', patch_data=self.patch_data)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )
