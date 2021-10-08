from torch.utils.data import Dataset
from glob import glob
import os
import pandas as pd


class ISC(Dataset):
    TRAIN_DIR = "training_images"
    QUERY_DIR = "query_images"
    REFERENCE_DIR = "reference_images"
    GT = "public_ground_truth.csv"

    def __init__(self, root, split="train", q_transform=None, k_transform=None):
        super().__init__()
        self.q_transform = q_transform
        self.k_transform = k_transform

        if split == "train":
            q_imgs = list(glob(os.path.join(root, self.TRAIN_DIR, "*.jpg")))
            k_imgs = q_imgs
        elif split == "query_val":
            df = pd.read_csv(root, self.GT)
            df.dropna(subset=["reference_id"], inplace=True)
            q_imgs = [os.path.join(root, self.QUERY_DIR, img) for img in df.iloc[:, 0]]
            k_imgs = [os.path.join(root, self.REFERENCE_DIR, img) for img in df.iloc[:, 1]]
        elif split == "query_all":
            q_imgs = list(glob(os.path.join(root, self.QUERY_DIR, "*.jpg")))
            k_imgs = q_imgs
        elif split == "reference_all":
            q_imgs = list(glob(os.path.join(root, self.REFERENCE_DIR, "*.jpg")))
            k_imgs = q_imgs
        else:
            raise NotImplementedError(f"split {split} is not implemented in ISC dataset")

        self.data = list(zip(q_imgs, k_imgs))

    def __getitem__(self, indx):
        q_img, k_img = self.data[indx]
        if self.q_transform is not None:
            q_img = self.q_transform(q_img)
        if self.k_transform is not None:
            k_img = self.k_transform(k_img)

        return q_img, k_img

    def __len__(self):
        return len(self.data)
