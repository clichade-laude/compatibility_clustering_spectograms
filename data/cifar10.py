import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset

from data.poison import PoisonDataset

import numpy as np
from os.path import join

# https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/trainer.py
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225])

train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
        #  transforms.RandomCrop(32, padding=4),
         transforms.Resize((128,128)),
         transforms.ToTensor(),
         normalize])

test_transform = transforms.Compose(
    [transforms.Resize((128,128)),
     transforms.ToTensor(),
     normalize])

def cifar10_loader(path, batch_size=128, train=True, oracle=False, augment=True, poison=True, dataset=None):
    ds_name = "spectrogram-dataset"

    if dataset is None:
        transform = train_transform if train and augment else test_transform

        if path == "clean": poison = False
        if oracle and train: poison = False
        path = path if poison else None

        split = "train" if train else "test"
        dataset = PoisonDataset(root=join('datasets', ds_name, split), train=train, transform=transform, poison_params=path)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=train and augment,
        num_workers=2, pin_memory=True)

    return dataset, dataloader

