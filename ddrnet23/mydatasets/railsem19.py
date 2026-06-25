import os
from typing import List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class RailSem19Dataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split_file: str,
        image_size=(1024, 512),
        augment=False,
        ignore_index=255,
    ):
        self.data_root = data_root
        self.img_dir = os.path.join(data_root, "jpgs", "rs19_val")
        self.lbl_dir = os.path.join(data_root, "uint8", "rs19_val")
        self.split_file = split_file
        self.image_size = image_size  # (W, H)
        self.augment = augment
        self.ignore_index = ignore_index

        with open(split_file, "r") as f:
            self.samples: List[str] = [x.strip() for x in f if x.strip()]

        self.img_exts = [".jpg", ".jpeg", ".png"]
        self.lbl_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

    def __len__(self):
        return len(self.samples)

    def _find_file(self, folder, stem, exts):
        for ext in exts:
            p = os.path.join(folder, stem + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Could not find file for stem={stem} in {folder}")

    def _load_pair(self, stem):
        img_path = self._find_file(self.img_dir, stem, self.img_exts)
        lbl_path = self._find_file(self.lbl_dir, stem, self.lbl_exts)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
        if label is None:
            raise RuntimeError(f"Failed to read label: {lbl_path}")

        if label.ndim == 3:
            label = label[:, :, 0]

        return image, label

    def _resize(self, image, label):
        w, h = self.image_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
        return image, label

    def _augment_fn(self, image, label):
        if np.random.rand() < 0.5:
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()
        return image, label

    def _normalize(self, image):
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, idx):
        stem = self.samples[idx]
        image, label = self._load_pair(stem)
        image, label = self._resize(image, label)

        if self.augment:
            image, label = self._augment_fn(image, label)

        image = self._normalize(image)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label.astype(np.int64))

        return {
            "image": image,
            "label": label,
            "name": stem,
        }
