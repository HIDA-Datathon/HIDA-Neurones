import glob
from PIL import Image
import json
import numpy as np
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch


class NeutronDataset(Dataset):
    def __init__(self, data, target, n_classes=21, transform=None):
        self.n_classes = n_classes
        self.data = data
        self.target = target
        self.transform = transform

    def get_one_hot(self, targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape) + [nb_classes])

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        y = np.moveaxis(self.get_one_hot(y.astype(int), self.n_classes), -1, 0)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

    def __len__(self):
        return len(self.data)


class NeutronDataLoader(pl.LightningDataModule):
    def __init__(self, data_dir: str = "C:/Users/Tobias/Downloads/HIDA-ufz_image_challenge/photos_annotated",
                 batch_size: int = 8,
                 num_workers: int = 1, transform=None):
        super().__init__()

        self.LABEL_SUFFIX = "*.png"
        self.IMAGE_SUFFIX = "*.jpg"

        self.channels = 3
        self.image_shape = None
        self.train_data = None
        self.valid_data = None
        self.test_data = None

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def load_data(self):
        images = sorted(glob.glob(os.path.join(self.data_dir, self.IMAGE_SUFFIX)))
        labels = sorted(glob.glob(os.path.join(self.data_dir, self.LABEL_SUFFIX)))

        image_array = []
        label_array = []

        for image_file, label_file in zip(images, labels):
            image_array.append(np.array(Image.open(image_file)))
            this_label = np.array(Image.open(label_file))
            if len(this_label.shape) == 2:
                label_array.append(this_label)
            elif len(this_label.shape) == 3:
                label_array.append(this_label[:, :, 0])
            else:
                print("Error")

        image_array = np.moveaxis(np.array(image_array), -1, 1)
        return image_array / 255, np.array(label_array)

    def setup(self, stage=None):

        image_array, label_array = self.load_data()

        self.image_shape = (image_array.shape[1], image_array.shape[2])

        length = image_array.shape[0]

        train_split_start = 0
        train_split_end = int(length * 0.8)
        valid_split_start = train_split_end
        valid_split_end = int(length * 0.9)
        test_split_start = valid_split_end
        test_split_end = length

        if stage == 'fit' or stage is None:
            self.train_data = NeutronDataset(image_array[train_split_start: train_split_end],
                                             label_array[train_split_start: train_split_end],
                                             transform=self.transform)
            self.valid_data = NeutronDataset(image_array[valid_split_start: valid_split_end],
                                             label_array[valid_split_start: valid_split_end],
                                             transform=self.transform)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_data = NeutronDataset(image_array[test_split_start: test_split_end],
                                            label_array[test_split_start: test_split_end],
                                            transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
