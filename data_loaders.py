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
import pdb

import torch
import torchvision
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


from tqdm import tqdm

classes = ('Beco', 'Connie', 'Hank', 'Jati',
           'MyThai', 'Pheobe', 'Rudy', 'Sabu', 'Schottzie', 'Sunny')


#### 
### Adopted from: https://github.com/ufoym/imbalanced-dataset-sampler/tree/master/examples
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        else:
            raise NotImplementedError
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


"""
Fetch dataloader - train and validation
"""
def fetch_dataloader(input_size,  batch_size, use_sampler = True, use_cuda=True):
    normalize = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    data_transforms = {
        'train': torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalize,
        ]),
     
        'val': torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize,
        ])

    }

    subset = {
            "train": datasets.ImageFolder(root='data_sep_mov/train',transform=data_transforms["train"]),
            "val": datasets.ImageFolder(root = "data_sep_mov/val", transform = data_transforms["val"])
    }

    
    ## Subset   
    if use_sampler == True:
        train_loader = torch.utils.data.DataLoader(subset["train"], batch_size=batch_size, sampler=ImbalancedDatasetSampler(subset["train"]), **kwargs)
        val_loader = torch.utils.data.DataLoader(subset["val"], batch_size=batch_size, shuffle=False, sampler=None, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(subset["train"], batch_size=batch_size, sampler=None, **kwargs)
        val_loader = torch.utils.data.DataLoader(subset["val"], batch_size=batch_size, shuffle=False, sampler=None, **kwargs)

    print('Dataset: %d training samples & %d val samples\n' % (
    len(train_loader.dataset), len(val_loader.dataset)))
    

    ## Save distribution of data
    visualize_distribution_of_dataset(train_loader,val_loader)
    return train_loader, val_loader


"""
Fetch test dataloader
"""
def fetch_test_dataloader( input_size, batch_size):
    data_transforms = transforms.Compose ( [
            transforms.Resize ( input_size ),
            transforms.CenterCrop ( input_size ),
            transforms.ToTensor (),
            transforms.Normalize ( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] )
        ] )

    test = datasets.ImageFolder(root='data_sep_mov/test',transform=data_transforms)
    dataloader = torch.utils.data.DataLoader (test, batch_size=batch_size,shuffle=False, num_workers=8,pin_memory=True )
    
    return dataloader


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



def visualize_distribution_of_dataset(train_loader, val_loader):
    print('Distribution of classes in trainset dataset:')
    fig, ax = plt.subplots(figsize=(10,4))
    labels = [label for _, label in train_loader.dataset.imgs]
    classe_labels, counts = np.unique(labels, return_counts=True)
    ax.bar([classes[i] for i in (list(classe_labels))], counts)
    ax.set_xticks(classe_labels)
    plt.savefig("distribution_train_set.png")

    fig, ax = plt.subplots(figsize=(10,4))
    labels = [label for _, label in val_loader.dataset.imgs]
    classe_labels, counts = np.unique(labels, return_counts=True)
    ax.bar([classes[i] for i in (list(classe_labels))], counts)
    ax.set_xticks(classe_labels)
    plt.savefig("distribution_val_set.png")


def visualize_distribution_of_test_dataset(test_loader):
    print('Distribution of classes in testset dataset:')
    fig, ax = plt.subplots(figsize=(10,4))
    labels = [label for _, label in test_loader.dataset.imgs]
    classe_labels, counts = np.unique(labels, return_counts=True)
    ax.bar([classes[i] for i in (list(classe_labels))], counts)
    ax.set_xticks(classe_labels)
    plt.savefig("distribution_test_set.png")


