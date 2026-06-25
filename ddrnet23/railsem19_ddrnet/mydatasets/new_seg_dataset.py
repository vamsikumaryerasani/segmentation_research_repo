import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class NewSegDataset(Dataset):
    def __init__(
        self,
        data_root,
        split_file,
        image_size=(1024, 512),
        augment=False,
        ignore_index=255,
    ):
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, "images")
        self.mask_dir = os.path.join(data_root, "masks")
        self.image_size = image_size
        self.augment = augment
        self.ignore_index = ignore_index

        with open(split_file, "r") as f:
            self.ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.ids)

    def _find_file(self, folder, sample_id):
        exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]
        for ext in exts:
            path = os.path.join(folder, sample_id + ext)
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"Could not find file for sample_id={sample_id} in {folder}")

    def _load_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {path}")

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        mask = mask.astype(np.int64)
        return mask

    def _resize(self, image, mask):
        w, h = self.image_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return image, mask

    def _maybe_augment(self, image, mask):
        if not self.augment:
            return image, mask

        if np.random.rand() < 0.5:
            image = np.ascontiguousarray(np.fliplr(image))
            mask = np.ascontiguousarray(np.fliplr(mask))

        return image, mask

    def __getitem__(self, idx):
        sample_id = self.ids[idx]

        img_path = self._find_file(self.image_dir, sample_id)
        mask_path = self._find_file(self.mask_dir, sample_id)

        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        if mask.ndim != 2:
            raise ValueError(f"Mask must be single-channel, got shape {mask.shape} for {mask_path}")

        unique_vals = np.unique(mask)
        bad_vals = unique_vals[(unique_vals < 0) | (unique_vals > 5)]
        if len(bad_vals) > 0:
            raise ValueError(
                f"Invalid label values in {mask_path}. "
                f"Found unique values {unique_vals.tolist()}, "
                f"but expected only 0..5."
            )

        image, mask = self._resize(image, mask)
        image, mask = self._maybe_augment(image, mask)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(mask).long()

        return {
            "image": image,
            "label": label,
            "img_path": img_path,
            "mask_path": mask_path,
            "id": sample_id,
        }
