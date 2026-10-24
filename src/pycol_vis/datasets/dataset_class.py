import torch
import cv2


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, preprocess, labels=None):
        self.image_paths = image_paths
        self.preprocess = preprocess
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f"Could not read image: {img_path}")

        img = self.preprocess(img)

        if self.labels is not None:
            return img, self.labels[idx]

        return img
