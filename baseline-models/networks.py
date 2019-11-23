import torch
import torch.nn as nn
from torchvision import models

import numpy as numpy


 
"""
Freeze all parameters of the model
"""
def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False



def initialize_model(model_name, num_classes, use_pretrained = True):
    model_ft = None
    input_size = 0

    if model_name=="resnet50":
        """
        Resnet50
        """
        model_ft = models.resnet50(pretrained = use_pretrained)
        set_parameter_requires_grad(model_ft)

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes )
        input_size = 224

        return model_ft, input_size
    elif model_name == "resnet34":
        """
        Resnet34
        """
        model_ft  = models.resnet34(pretrained = use_pretrained)
        set_parameter_requires_grad(model_ft)

        num_ftrs =model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)

        input_size = 224

    elif model_name =="resnet101":
        """
        Resnet101
        """
        model_ft = models.resnet101(pretrained = use_pretrained)
        set_parameter_requires_grad(model_ft)
        
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

        input_size = 224

    elif model_name == "densenet":
            
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft)

        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    return model_ft, input_size

