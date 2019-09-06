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

import sklearn.metrics as sm

import pandas as pd

def resume_training():
    start_epoch, all_epoch_train_losses, all_epoch_val_losses,all_epoch_train_accuracy_prec1, all_epoch_train_accuracy_prec5,all_epoch_val_accuracy_prec1, all_epoch_val_accuracy_prec5 = load_last_model(model,model_path)


    best_acc = -9999
    best_acc_5 = -9999
    since = time.time()
    best_epoch = 0

    ## confusion matrix
    confusion_mtxes = []
    for epoch in (range(start_epoch+1, start_epoch+epochs+1)):

        print('Epoch {}/{}'.format(epoch, start_epoch+epochs))
        print('-' * 10)

        # train for one epoch
        (train_loss, train_acc1, train_acc5)  = train(train_loader, model, criterion, optimizer,exp_lr_scheduler)

        # evaluate on validation set
        (val_loss, val_acc1, val_acc5, val_confusion_matrix)   = evaluate(val_loader, model, criterion)


        all_epoch_train_losses.append(train_loss)
        all_epoch_train_accuracy_prec1.append(train_acc1)
        all_epoch_train_accuracy_prec5.append(train_acc5)
        
        
        all_epoch_val_losses.append(val_loss)
        all_epoch_val_accuracy_prec1.append(val_acc1)
        all_epoch_val_accuracy_prec5.append(val_acc5)
        
        confusion_mtxes.append(val_confusion_matrix)

        # deep copy the model
        if  val_acc1 > best_acc:
            best_acc = val_acc1
            best_acc_5 = val_acc5
            best_model_wts = model.state_dict()
            best_epoch = epoch
            save_checkpoint(model, model_path, epoch, train_loss, val_loss, all_epoch_train_losses,all_epoch_val_losses, all_epoch_train_accuracy_prec1,all_epoch_train_accuracy_prec5,all_epoch_val_accuracy_prec1, all_epoch_val_accuracy_prec5)

        print("Train and validation complete")
        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch,
                train_loss, train_acc1))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch,
                val_loss, val_acc1))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}, {:4f}'.format(best_acc,best_acc_5))
    print("Best epoch,", best_epoch)
    
    # load best model weights
    model.load_state_dict(best_model_wts)

    visualize_save_plots(all_epoch_train_losses,all_epoch_val_losses, all_epoch_train_accuracy_prec1,
    all_epoch_val_accuracy_prec1,confusion_mtxes,classes)

    return model


def train(train_loader, model, criterion, optimizer,scheduler):
    total_images = len(train_loader)

    ## set model to train
    model.train(True)
    
    ## initialize values
    losses = RunningAverage("train loss")
    top1 = RunningAverage("train top1")
    top5 = RunningAverage("train top5")

    
    for (inputs, labels) in tqdm(train_loader):
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        ## zero the parameter gradients
        optimizer.zero_grad()

        ## forward
        outputs = model(inputs)

        loss = criterion(outputs, labels)

        batch_size = inputs.size(0)
        (acc1, acc5) = accuracy(outputs.data.cpu(), labels.data.cpu(),topk = (1,5))

        ## propagate loss backward
        loss.backward()
        optimizer.step()
        scheduler.step()

        ## statistics
        losses.update(loss.item(),batch_size)
        top1.update(acc1[0], batch_size)
        top5.update(acc5[0], batch_size)

    return (losses.avg, top1.avg, top5.avg)


def evaluate(val_loader, model, criterion):
    model.eval()

    total_images = len(val_loader)
    
    ## initialize values
    losses = RunningAverage("val loss")
    top1 = RunningAverage("val top1")
    top5 = RunningAverage("val top5")

    ## confusion matrix = during training
    correct = 0
    targets, preds = [], []

    with torch.no_grad():
        for  (images, target) in tqdm(val_loader):
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

    return (losses.avg, top1.avg, top5.avg,confusion_mtx)

## Compute accuracy of the model
##
## Now, in the case of top-1 score, - top class (the one having the highest probability) is the same as the target label.
# In the case of top-5 score, - target label is one of your top 5 predictions (the 5 ones with the highest probabilities).
##
##
########
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


#================================================================================================================================
# Parse arguments
# 
# 
def str2bool(s):
    if s.lower() in ('yes','true','t','y',1):
        return True
    elif s.lower() in ('no','false','f','n',0):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")


parser = argparse.ArgumentParser(description = "Asian Elephant Facial Recognition Model")
parser.add_argument('--batch_size', type=int, default = 1, metavar='N', help = "input batch size for training (default : 1)")
parser.add_argument("--epochs",type=int, default = 2000, metavar = 'N', help="number of epochs to train the model (default: 2000)") 
parser.add_argument("--no-cuda",action='store_true', default=True, help="enable CUDA training (default: True)")
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--device_id', type=int, default=0, help='gpu device id (default: 1)')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate for the adam optimizer')
parser.add_argument('--data_path', default='None',
                    help='data directory for train, val, and test. the root dir is in data/')
parser.add_argument('--output_path', default='None',
                    help='directory to save learned representation')
parser.add_argument('--model_path', default='None',
                    help='directory to save models and losses. root dir is in models/')
parser.add_argument('--epoch_save_interval', type=int, default=5,
                    help='model that would be saved after every given interval e.g.,  250')
parser.add_argument('--model_name', default='None',
                    help='trained model name e.g., used during evaluation stage')

parser.add_argument('--use_sampler', type=str2bool, nargs='?', const=True,
                    help='boolean variable indicating if we are using sampler for training')

args = parser.parse_args()
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
data_path = args.data_path
output_path 		= args.output_path
model_path 		= args.model_path
epoch_save_interval 	= args.epoch_save_interval
model_name 		= args.model_name
use_sampler = args.use_sampler


## visualize the distribution of classes in train and validation loader
classes = ["Beco", "Connie", "Hank",  
            "Jati",  "Kamala",  "Maharani",  "MyThai", "Pheobe" ,"Rudy", "Sabu", 
            "Schottzie", "Spike", 
            "Sunny", "Swarna"]

num_classes = len(classes)

def adjust_learning_rate(optimizer, epoch,lr):
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
## Create model directory if it doesnt exists, for saving the trained models
def create_dir(path, text):
    directory = Path(path)
    if not directory.exists():
        os.makedirs(directory)
    else:
        print(text.format(directory))
    return 

## Create model and output directory
create_dir(model_path,"model directory exists: {}")
create_dir(output_path + "_train", "output directory exists: {}")



### initialize global variables
args.cuda = True
device_id = args.device_id
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Initialize and select the pretrained model
 ## enable logging
set_logger ( os.path.join ( model_path, 'train.log' ))
use_dropout = False

(model, input_size) = nnet.initialize_model(model_name,num_classes=num_classes,use_pretrained=True)

# enable cudalabels
if (torch.cuda.is_available()):
    is_cuda = True
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
else:
	is_cuda = False

logging.info("Initialized the model")
use_cuda = args.cuda

## SGD optimizer, can try Adam optimizer
### Get data
# Only parameters of final layer are being optimized as
if torch.cuda.device_count()>1:
    optimizer 	= torch.optim.SGD(model.module.fc.parameters(), lr=lr, momentum=0.9) if model_name=="resnet" else torch.optim.SGD(model.module.classifier.parameters(), lr = lr)
else:
    optimizer 	= torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9) if model_name=="resnet" else torch.optim.SGD(model.classifier.parameters(), lr = lr)


# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

### Dataloaders
train_loader, val_loader, train_weights = data_loaders.fetch_dataloader(data_path,input_size,batch_size, use_sampler, use_cuda)
print(train_weights, len(train_weights))
train_weights = train_weights.type(torch.FloatTensor)
train_weights = torch.FloatTensor(train_weights) if not use_cuda else torch.FloatTensor(train_weights).cuda()

## Initialize the loss function
if not use_sampler:
    if use_cuda:
        criterion  = torch.nn.CrossEntropyLoss(weight=train_weights).cuda()
    else:
        criterion  = torch.nn.CrossEntropyLoss(weight=train_weights)
else:
    if use_cuda:
        criterion  = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion  = torch.nn.CrossEntropyLoss().cuda()


## visualize some images:
#mages, labels = next(iter(train_loader))
#data_loaders.make_grid(images,labels,classes)

#data_loaders.visualize_distribution_of_dataset(train_loader, val_loader,classes)
## Start training
logging.info("Finished fetching dataloader for train and val: {}, {}".format(len(train_loader),len(val_loader)))


resume_training()	




