## resnet34_ModelEpoch_25_Train_loss_2.7302_Val_loss_2.8225.pth

import argparse
import logging
import warnings
import numpy as np
import torch
import os
import random
import scipy.io
from pathlib import Path
import time
import pdb
import os.path as osp
import os
import scipy.io as sio
import argparse
from glob import glob
import torch.nn as nn
from torch.backends import cudnn
from torch.optim.lr_scheduler import StepLR
from torch.optim import lr_scheduler

import data_loaders

import networks as nnet

from torch.autograd import Variable

from running_average import RunningAverage
from utils import set_logger, save_dict_to_json, save_checkpoint, load_last_model,visualize_save_plots,visualize_test_acc
from torch.utils.model_zoo import tqdm
from opts import test_opts  # The options for the project
import sklearn.metrics as sm

import pandas as pd

def accuracy(output, target, topk=(1,) ):
    with torch.no_grad ():
        maxk = max ( topk )
        batch_size = target.size ( 0 )
        _, pred = output.topk ( maxk, 1, True, True )
        pred = pred.t()
        correct = pred.eq ( target.view ( 1, -1 ).expand_as ( pred ) )

    res = []
    for k in topk:
        correct_k = correct[:k].view ( -1 ).float ().sum ( 0, keepdim=True )
        res.append ( correct_k.mul_ ( 100.0 / batch_size ) )
    
    return res

## Read Args
args = test_opts()
model_path = args.model_path
data_path = args.data_path
batch_size = args.batch_size
output_path = args.output_path
## Load Data
file_name_classes = args.csv_classes
with open(file_name_classes,'r') as f:
    classes = f.readlines()

num_classes = len(classes)


## Get Data
input_size = 224
dataloader = data_loaders.fetch_test_dataloader( data_path, input_size, batch_size,output_path)

## Initialize model
(model, input_size) = nnet.initialize_model("resnet34",num_classes=num_classes,use_pretrained=True)
# enable cuda labels
if (torch.cuda.is_available()):
    use_cuda = True
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.cuda()
else:
	use_cuda = False


# load model weights
model.load_state_dict(torch.load(model_path))
print("Loaded model",model)

if use_cuda:
    criterion  = torch.nn.CrossEntropyLoss().cuda()
else:
    criterion  = torch.nn.CrossEntropyLoss()

model.eval()

total_images = len(dataloader)
    
## initialize values
losses = RunningAverage("test loss")
top1 = RunningAverage("test top1")
top5 = RunningAverage("test top5")

## confusion matrix = during training
correct = 0
targets, preds = [], []
with torch.no_grad():
    for  (images, target) in tqdm(dataloader):
        if use_cuda:
            images = images.cuda()
            target = target.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data.cpu(), target.data.cpu(), topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))


        ## build confusion matrix
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        targets += list(target.cpu().numpy())
        preds += list(pred.cpu().numpy())
        confusion_mtx = sm.confusion_matrix(targets, preds)

print('Test Acc: {:4f}, {:4f}'.format(top1.avg, top5.avg))