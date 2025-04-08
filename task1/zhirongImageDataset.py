import os
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, data_dir, mode='train', transforms=None, split_ratio=0.99):
        A_dir = os.path.join(data_dir, 'VAE_generation/train') # modification forbidden
        B_dir = os.path.join(data_dir, 'VAE_generation_Cartoon/train')  # modification forbidden

        files_A = sorted(os.listdir(A_dir))
        files_B = sorted(os.listdir(B_dir))

        split_idx_A = int(len(files_A) * split_ratio)
        split_idx_B = int(len(files_B) * split_ratio)

        if mode == 'train':
            self.files_A = [os.path.join(A_dir, name) for name in files_A[:split_idx_A]]
            self.files_B = [os.path.join(B_dir, name) for name in files_B[:split_idx_B]]
        elif mode == 'valid':
            self.files_A = [os.path.join(A_dir, name) for name in files_A[split_idx_A:]]
            self.files_B = [os.path.join(B_dir, name) for name in files_B[split_idx_B:]]

        self.transforms = transforms

    def __len__(self):
        return len(self.files_A)

    def __getitem__(self, index):
        file_A = self.files_A[index]
        file_B = self.files_B[index]

        img_A = Image.open(file_A)
        img_B = Image.open(file_B)

        if self.transforms is not None:
            img_A = self.transforms(img_A)
            img_B = self.transforms(img_B)

        return img_A, img_B
