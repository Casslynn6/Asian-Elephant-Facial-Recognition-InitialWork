import torch
import torch.nn as nn
from torchvision import models

import numpy as numpy


classes = ('Beco', 'Connie', 'Hank', 'Jati',
           'MyThai', 'Pheobe', 'Rudy', 'Sabu', 'Schottzie', 'Sunny')



"""
Freeze all parameters of the model
"""
def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


"""
Loss function
"""

def loss_fn(outputs, labels):
    """
    Compute cross entropy loss given outputs and labels

    """
    return nn.CrossEntropyLoss()(outputs, labels)


def initialize_model(model_name, num_classes, use_pretrained = True):
    model_ft = None
    input_size = 0


    """
    Resnet50
    """
    model_ft = models.resnet50(pretrained = use_pretrained)
    set_parameter_requires_grad(model_ft)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes )
    input_size = 224

    return model_ft, input_size

