from albumentations.augmentations.transforms import ChannelShuffle
import torch
import pandas as pd
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import albumentations as A 
from albumentations.pytorch import ToTensorV2
import os
import sys
import tqdm

SEED_VALUE = 0

def split_data(root_dir = "./dataset/dog_cats",
    split_size = 0.2,
    override_saved_file = True
    ):

    #create csv with image name and class
    files_list = glob(root_dir + '/*.jpg')
    labels = list(map(lambda x: 0 if "cat" in os.path.basename(x) else 1, files_list))

    data = pd.DataFrame()
    data["file_name"] = files_list
    data['labels'] = labels

    if override_saved_file:
        data.to_csv("all_data.csv")

    # split the data
    train, test = train_test_split(data, stratify=data['labels'], random_state=SEED_VALUE, test_size=split_size)

    return train, test


class ImageFolder(nn.Module):
    def __init__(self, df, transform = None):
        super(ImageFolder, self).__init__()
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        img_path, label = self.df.iloc[index, 0], self.df.iloc[index, 1]
        image = np.array(Image.open(img_path))

        if self.transform is not None:
            augmentations = self.transform(image = image)
            image = augmentations["image"]

        return image, label

train_transform = A.Compose([
    A.Resize(width=224, height = 224),
    #A.RandomCrop(width=224, height = 224),
    A.Rotate(limit = 40, p = 0.9, border_mode=cv2.BORDER_CONSTANT),
    A.HorizontalFlip(p = 0.5),
    A.VerticalFlip(p = 0.1),
    A.ChannelShuffle(p = 0.5),
    A.GaussNoise(var_limit=(10.0, 30.0), p = 0.3),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p = 0.9),
    A.OneOf([
        A.Blur(blur_limit = 3, p = 0.5),
        A.ColorJitter(p = 0.5),
    ], p = 1.0),


    #ToTensor --> Normalize(mean, std) 
    A.Normalize(
        mean = [0.4194, 0.4042, 0.3910],
        std = [0.2439, 0.2402, 0.2372],
        max_pixel_value = 255,
    ),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Resize(width=224, height = 224),
    #ToTensor --> Normalize(mean, std) 
    A.Normalize(
        mean = [0.4194, 0.4042, 0.3910],
        std = [0.2439, 0.2402, 0.2372],
        max_pixel_value = 255,
    ),
    ToTensorV2()
])


def get_dataset(root_dir = "./dataset/dog_cats", 
    train_transform = train_transform,
    test_transform = test_transform,
    split_size = 0.2
    ):

    train, test = split_data(root_dir = "../dataset/dog_cats",
                        split_size = split_size)

    train_dataset = ImageFolder(df=train, transform=train_transform)
    test_dataset = ImageFolder(df=test, transform=test_transform)

    dataset = {}
    dataset["train"], dataset["test"] = train_dataset, test_dataset
    return dataset

def get_mean_std():
    dataset = get_dataset()

    train_loader = DataLoader(dataset["train"], batch_size=1000)
    #test_loader = DataLoader(dataset["test"], batch_size=1000)

    mean = 0.
    std = 0.
    for images, _ in tqdm.tqdm(train_loader):
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(train_loader.dataset)
    std /= len(train_loader.dataset)

    print(mean)  #tensor([0.4194, 0.4042, 0.3910])
    print(std) #tensor([0.2439, 0.2402, 0.2372])
