# Ignore warnings
import warnings

from torch.utils.data import DataLoader, Dataset

#import data_utils

warnings.filterwarnings ( "ignore" )

from torchvision import transforms, datasets
import glob
import os
import numpy as np
import ntpath
from skimage import io

import torch
import torchvision
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from tqdm import tqdm
"""
Fetch dataloader - train and validation
"""
def fetch_dataloader(input_size,  batch_size):
    data_transforms = {
        'train': transforms.Compose ( [
            transforms.RandomResizedCrop ( input_size ),
            transforms.RandomHorizontalFlip (),
            transforms.ToTensor (),
            transforms.Normalize ( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] )
        ] ),
        'val': transforms.Compose ( [
            transforms.Resize ( input_size ),
            transforms.CenterCrop ( input_size ),
            transforms.ToTensor (),
            transforms.Normalize ( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] )
        ] ),
    }

    subset = {
    "train": datasets.ImageFolder(root='data/train',transform=data_transforms["train"]),
    "val": datasets.ImageFolder(root = "data/val", transform = data_transforms["val"])
    }

    
    print(len(subset["train"]))    
 
    ## Subset   
    dataloaders_dict = {x: torch.utils.data.DataLoader ( subset[x], batch_size=batch_size,
                                                         shuffle=True, num_workers=4 )
                        for x in ['train', 'val']}


    return dataloaders_dict


"""
Fetch test dataloader
"""
def fetch_test_dataloader( input_size, batch_size):
    data_transforms = transforms.Compose ( [
        transforms.RandomResizedCrop ( input_size ),
        transforms.RandomHorizontalFlip (),
        transforms.ToTensor (),
        transforms.Normalize ( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] )
    ] )

    testset = datasets.ImageFolder(root='data/test',transform=data_transforms),

    testloader = torch.utils.data.DataLoader ( testset, batch_size=batch_size, shuffle=False )
    return testloader


"""
Test images
"""
def imshow(img):
    img = img/2 + 0.5 ## unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

def make_grid(images,labels, classes):
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))
