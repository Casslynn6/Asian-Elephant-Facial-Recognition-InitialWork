import os
import json
import shutil
import torch.optim
from torch.autograd import Variable
import torch.nn as nn
import random
import numpy as np
import torch.backends.cudnn as cudnn

from models.resnet import resnet  # The model construction
from utils import opts, get_config

best_prec1 = 0

def main():
    global arg, best_prec1

    args = opts()

    ## 
    model_source, model_target = resnet(args)

    
