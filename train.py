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

from torch.backends import cudnn
from torch.optim.lr_scheduler import StepLR

import data_loaders
import networks as nnet

from torch.autograd import Variable

from running_average import RunningAverage
from utils import set_logger, save_dict_to_json, save_checkpoint, load_last_model
from torch.utils.model_zoo import tqdm


def resume_training():
    start_epoch, all_epoch_train_losses, all_epoch_val_losses,all_epoch_train_accuracy_prec1, all_epoch_train_accuracy_prec5,all_epoch_val_accuracy_prec1, all_epoch_val_accuracy_prec5 = load_last_model(model,model_path)
    
    for epoch in (range(start_epoch+1, start_epoch+epochs+1)):

        print('Epoch {}/{}'.format(epoch, start_epoch+epochs))
        print('-' * 10)

        ## train and evaluate the model
        for phase in ['train','val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)
                model.eval()

            running_loss_train = 0.0
            running_loss_val = 0.0

            running_corrects_train_prec1 = 0
            running_corrects_train_prec5 = 0
            running_correct_val_prec1 = 0
            running_correct_val_prec5 = 0

            ##
            for data in tqdm(dataloaders_dict[phase]):
                ## inputs and labels
                inputs, labels = data

                if is_cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                
                optimizer.zero_grad()

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                ## No of samples
                n_samples = inputs.size ()[0]
                (prec1, prec5) = accuracy(outputs.data.cpu(), labels.data.cpu(),topk=(1,5))

                #pdb.set_trace()
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                    ## statistics
                    running_loss_train += loss.data[0]
                    running_corrects_train_prec1+=prec1
                    running_corrects_train_prec5+=prec5
                else:
                     # statistics
                    running_loss_val+=loss.data[0]
                    running_correct_val_prec1 +=prec1
                    running_correct_val_prec5 +=prec5

                
            epoch_loss = running_loss_train / len(dataloaders_dict[phase]) if phase == "train"  else running_loss_val / len(dataloaders_dict[phase]) 
            epoch_acc_prec1 =  running_corrects_train_prec1 / len(dataloaders_dict[phase]) if phase == "train"  else running_corrects_val_prec1 / len(dataloaders_dict[phase]) 
            epoch_acc_prec5 =  running_corrects_train_prec5/ len(dataloaders_dict[phase]) if phase == "train"  else running_corrects_val_prec5 / len(dataloaders_dict[phase])


            logging.info('{}  Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == "train":
                all_epoch_train_losses.append(epoch_loss)
                all_epoch_train_accuracy_prec1.append(epoch_acc_prec1)
                all_epoch_train_accuracy_prec5.append(epoch_acc_prec5)
                train_loss = epoch_loss
            else:
                all_epoch_val_losses.append(epoch_loss)
                all_epoch_val_accuracy_prec1.append(epoch_acc_prec1)
                all_epoch_val_accuracy_prec5.append(epoch_acc_prec5)
                val_loss = epoch_loss
        
            # deep copy the model
            if phase == 'val' and epoch_acc_prec1 > best_acc:
                best_acc = epoch_acc_prec1
                best_acc_5 = epoch_acc_prec5
                best_model_wts = model.state_dict()
                save_checkpoint(model, model_path, epoch, train_loss, val_loss, all_epoch_train_losses,all_epoch_val_losses, all_epoch_train_accuracy_prec1,all_epoch_train_accuracy_prec5,all_epoch_val_accuracy_prec1, all_epoch_val_accuracy_prec5)


        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}, {:4f}'.format(best_acc,best_acc_5))

        # load best model weights
        model.load_state_dict(best_model_wts)

        return model


## COmpute accuracy of the model
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
parser.add_argument('--lr', type=float, default=0.000001,
                    help='learning rate for the adam optimizer')
parser.add_argument('--data_path', default='None',
                    help='data directory for train, val, and test. the root dir is in data/')
parser.add_argument('--output_path', default='None',
                    help='directory to save learned representation')
parser.add_argument('--model_path', default='None',
                    help='directory to save models and losses. root dir is in models/')
parser.add_argument('--epoch_save_interval', type=int, default=5,
                    help='model that would be saved after every given interval e.g.,  250')
parser.add_argument('--is_train', type=str2bool, nargs='?', const=True,
                    help='boolean variable indicating if we are training the model. for testing just disable this flag')
parser.add_argument('--model_name', default='None',
                    help='trained model name e.g., used during evaluation stage')
parser.add_argument('--eval_set', default='test',
                    help='which set to evaluate the model when is_train=False')


args = parser.parse_args()
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
data_path = args.data_path
output_path 		= args.output_path
model_path 		= args.model_path
epoch_save_interval 	= args.epoch_save_interval
is_train 	 	= args.is_train
eval_set 		= args.eval_set
model_name 		= args.model_name

pdb.set_trace()

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
create_dir(output_path + eval_set, "output directory exists: {}")


### initialize global variables
args.cuda = True
device_id = args.device_id
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers':1, 'pin_memory': True} if args.cuda else {}

## Initialize and select the pretrained model
 ## enable logging
set_logger ( os.path.join ( model_path, 'train.log' ))
use_dropout = False

(model, input_size) = nnet.initialize_model("resnet",num_classes=10,use_pretrained=True)
# enable cudalabels
if (torch.cuda.is_available()):
	model = model.cuda()
	torch.cuda.set_device(device_id)
	is_cuda = True
else:
	is_cuda = False
logging.info("Initialized the model")

## Initialize hyperparameters
# initialize the training hyper-parameters
momentum 	= 0.9
lr_step_size 	= epochs
criterion  	= nnet.loss_fn ## cross entropy loss
## SGD optimizer, can try Adam optimizer
optimizer 	= torch.optim.SGD(model.fc.parameters(), lr = lr)
scheduler 	= torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.1, last_epoch=-1)

logging.info ( "Starting training for {} epoch(s)".format ( epochs ) )


## Train and evaluate
is_train = True
if (is_train == True):	
    dataloaders_dict 	= data_loaders.fetch_dataloader(input_size = input_size, batch_size = batch_size)
    trainloader = dataloaders_dict["train"]
    valloader = dataloaders_dict["val"]
    logging.info("Finished fetching dataloader for train and val: {}, {}".format(len(trainloader),len(valloader)))
    resume_training()	
else:
    testloader = data_loaders.fetch_test_dataloader(input_size = input_size,batch_size = batch_size)


