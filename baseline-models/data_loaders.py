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

        total_count = len(dataset)
        self.label_weights = [1 - (1.0 * v/total_count) for k, v in label_to_count.items()]
        def normalize(probs):
            prob_factor = 1 / sum(probs)
            return [prob_factor * p for p in probs]

        self.label_weights = torch.DoubleTensor(normalize(self.label_weights))
        
        
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
def fetch_dataloader(data_path, input_size,  batch_size, save_image_path,  use_sampler = True, use_cuda=True):
    normalize = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    
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
            "train": datasets.ImageFolder(root=os.path.join(data_path,"train"),transform=data_transforms["train"]),
            "val": datasets.ImageFolder(root = os.path.join(data_path,"val"), transform = data_transforms["val"])
    }

    train_sampler = ImbalancedDatasetSampler(subset["train"])
    

    ## Subset   
    if use_sampler == True:
        train_loader = torch.utils.data.DataLoader(subset["train"], batch_size=batch_size, sampler=ImbalancedDatasetSampler(subset["train"]), **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(subset["train"], batch_size=batch_size, sampler=None, **kwargs)
    
    val_loader = torch.utils.data.DataLoader(subset["val"], batch_size=batch_size, shuffle=False, sampler=None, **kwargs)

    print('Dataset: %d training samples & %d val samples\n' % (
    len(train_loader.dataset), len(val_loader.dataset)))


    class_names = subset['train'].classes
    print(" Dataset contains following classes",",".join(class_names))

    #visualize_distribution_of_dataset(train_loader,val_loader,class_names,save_image_path)
    #visualize_images(train_loader,val_loader,class_names,save_image_path)
    
    return train_loader, val_loader


"""
Fetch test dataloader
"""
def fetch_test_dataloader( data_path, input_size, batch_size,save_image_path):
    data_transforms = transforms.Compose ( [
            transforms.Resize ( input_size ),
            transforms.CenterCrop ( input_size ),
            transforms.ToTensor (),
            transforms.Normalize ( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] )
        ] )

    test = datasets.ImageFolder(root=data_path,transform=data_transforms)
    dataloader = torch.utils.data.DataLoader (test, batch_size=batch_size,
    shuffle=False, num_workers=4,pin_memory=True )
    class_names = test.classes
    print(class_names)

    visualize_distribution_of_testset(dataloader,class_names,save_image_path)
    return dataloader


def visualize_images(train_loader, val_loader, class_names, save_image_path):

    def imsave(inp, save_path, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imsave(save_path,inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    # Get a batch of training data
    inputs, classes = next(iter(train_loader))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imsave(out, os.path.join(save_image_path, "train_images.png"), title=[class_names[x] for x in classes])

    # validation batch
    # Get a batch of training data
    inputs_v, classes_v = next(iter(val_loader))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs_v)
    imsave(out, os.path.join(save_image_path, "val_images.png"), title=[class_names[x] for x in classes_v])


def visualize_distribution_of_dataset(train_loader, val_loader, classes,save_path):

    print('Distribution of classes in trainset dataset:')
    fig, (ax1, ax2) = plt.subplots(2, figsize=(40,20))
    
    ## Train
    train_labels = [label for _, label in train_loader.dataset.imgs]
    classe_labels, counts = np.unique(train_labels, return_counts=True)
    
    
    ax1.bar([classes[i] for i in (list(classe_labels))], counts)
    ax1.set_xticks(classe_labels)
    ax1.set_title("Training set distribution")


    ## Validation
    val_labels = [label for _, label in val_loader.dataset.imgs]
    val_class_labels, counts = np.unique(val_labels, return_counts=True)
    ax2.bar([classes[i] for i in (list(val_class_labels))], counts)
    ax2.set_xticks(val_class_labels)
    ax2.set_title("Validation set distribution")

    
    plt.savefig(os.path.join(save_path,"distribution_train_val.png"))


def visualize_distribution_of_testset(test_loader, classes,save_path):
    print('Distribution of classes in testset dataset:')
    print(classes)
    fig, ax = plt.subplots(figsize=(10,4))
    labels = [label for _, label in test_loader.dataset.imgs]
    classe_labels, counts = np.unique(labels, return_counts=True)
    ax.bar([classes[i] for i in (list(classe_labels))], counts)
    ax.set_xticks(classe_labels)
    plt.savefig(os.path.join(save_path,"distribution_test_set.png"))


