from pathlib import Path
from typing import Optional, Tuple, Literal, List
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import random
import torch

class _SingleDataset(Dataset):
    def __init__(
        self,
        file_paths: List[str],
    ):
        self.file_paths = file_paths
        self.length = len(self.file_paths)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.file_paths[index])
        img = (transforms.ToTensor()(img)*255).type(torch.uint8)
        return img


class CustomImageDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        mode: str = 'train',
        transform: Optional[transforms.Compose] = None,
        val_split: float = 0.2,
    ):
        A_dir = os.path.join(data_dir, 'VAE_generation/train') # modification forbidden
        B_dir = os.path.join(data_dir, 'VAE_generation_Cartoon/train') # modification forbidden

        # Get and pair files before shuffling
        self.pairs = self._get_paired_files(A_dir, B_dir)

        # Split datasets while maintaining pairs
        split_idx = int(len(self.pairs) * val_split)
        if mode == 'train':
            self.pairs = self.pairs[split_idx:]
        elif mode == 'valid':
            self.pairs = self.pairs[:split_idx]
        else:
            raise ValueError("Invalid mode. Use 'train' or 'valid'")

        self.transform = transform
        self.length = len(self.pairs)

    def _get_paired_files(self, a_dir, b_dir):
        """Pair files by matching filenames (without extension)"""
        a_files = {os.path.splitext(f)[0]: f for f in os.listdir(a_dir)}
        b_files = {os.path.splitext(f)[0]: f for f in os.listdir(b_dir)}
        
        common_keys = sorted(a_files.keys() & b_files.keys())
        return [
            (os.path.join(a_dir, a_files[k]), os.path.join(b_dir, b_files[k]))
            for k in common_keys
        ]
    
    def get_partial_dataset(self, domain: Literal['A', 'B']):
        return _SingleDataset([path[0] if domain == 'A' else path[1] for path in self.pairs])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        a_path, b_path = self.pairs[index]
        
        img_A = Image.open(a_path)
        img_B = Image.open(b_path)

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return img_A, img_B, a_path, b_path