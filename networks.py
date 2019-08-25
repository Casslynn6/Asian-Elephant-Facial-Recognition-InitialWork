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

def weighted_loss_fn(outputs, labels):
    """
    Compute cross entropy loss given outputs and labels

    """
    ##{'Beco': 0.013, 'Connie': 0.009, 'Hank': 0.323, 'Jati': 0.134, 'MyThai': 0.189, 'Pheobe': 0.006, 'Rudy': 0.01, 'Sabu': 0.161, 'Schottzie': 0.138, 'Sunny': 0.018}
    weights = [0.0009165902841429881, 0.0005688282138794084, 0.0005076142131979696, 0.002631578947368421, .0034129692832764505,0.0004623208506703652, 0.000655307994757536, 0.0029585798816568047, 0.002617801047120419, 0.0011641443538998836]
    class_weights = torch.FloatTensor(weights).cuda()
    return nn.CrossEntropyLoss(weight=class_weights)(outputs, labels)


def loss_fn(outputs, labels):
    """
    Compute cross entropy loss given outputs and labels

    """
    return nn.CrossEntropyLoss(outputs, labels)



def initialize_model(model_name, num_classes, use_pretrained = True):
    model_ft = None
    input_size = 0


 
  
    if model_name=="resnet":
        """
        Resnet50
        """
        model_ft = models.resnet50(pretrained = use_pretrained)
        set_parameter_requires_grad(model_ft)

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes )
        input_size = 224

        return model_ft, input_size

    elif model_name == "densenet":
            
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft)

        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

        return model_ft, input_size

