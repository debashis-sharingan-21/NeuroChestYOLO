

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pandas as pd
import numpy as np

# Standard NIH-14 Pathology Classes + No Finding
NIH_15_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
    "No Finding"
]

def get_transforms(img_size=224, is_train=True):
    """
    Returns augmentation pipeline for training or standard normalization for validation/test.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])


class CXRDataset(Dataset):
    """
    PyTorch Dataset for multi-label chest radiograph classification.
    Supports either:
    1. Pre-encoded binary columns for each target class in CSV.
    2. Pipe-separated disease strings (e.g. 'Cardiomegaly|Effusion') in a label column.
    """
    def __init__(
        self,
        df_or_csv_path,
        images_dir,
        label_cols=None,
        img_col="image_path",
        finding_col=None,
        transform=None,
        img_size=224,
        is_train=False
    ):
        if isinstance(df_or_csv_path, str):
            self.df = pd.read_csv(df_or_csv_path)
        else:
            self.df = df_or_csv_path.copy()

        self.images_dir = images_dir
        self.img_col = img_col
        self.label_cols = label_cols or NIH_15_CLASSES
        self.num_classes = len(self.label_cols)
        self.transform = transform or get_transforms(img_size=img_size, is_train=is_train)

        # Handle pipe-separated string format if specified
        if finding_col and finding_col in self.df.columns:
            for label in self.label_cols:
                if label not in self.df.columns:
                    # Match exact label or replace space with underscore
                    pattern = label.replace("_", " ")
                    self.df[label] = self.df[finding_col].apply(
                        lambda s: 1.0 if isinstance(s, str) and (pattern in s or label in s) else 0.0
                    )

        # Precompute binary target tensor for fast access
        for col in self.label_cols:
            if col not in self.df.columns:
                self.df[col] = 0.0

        self.targets = self.df[self.label_cols].values.astype(np.float32)
        self.image_filenames = self.df[self.img_col].astype(str).tolist()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        img_path = filename if os.path.isabs(filename) else os.path.join(self.images_dir, filename)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at path: {img_path}")

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return img, target
