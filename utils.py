import json
import logging
import os
import shutil
import numpy as np
import torch

from glob import glob
from pathlib import Path
import pdb
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
"""
Save dictionary to json file
"""
def save_dict_to_json(d,json_path):
    with open(json_path,'w') as f:
        items = d
        json_dic = {}

        for k,v in items.items():
            if isinstance(v, dict):
                json_dic[k] = {}
                for i , j in v.items():
                    json_dic[k][i] = float(j)
            else:
                json_dic[k] = float(v)

        json.dump(json_dic,f,indent=4)


"""
Set logging
"""
def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        ## Log to file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter ( logging.Formatter ( '%(asctime)s:%(levelname)s: %(message)s' ) )
        logger.addHandler(file_handler)

    ## stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter ( logging.Formatter ( '%(asctime)s:%(levelname)s: %(message)s' ) )
    logger.addHandler(stream_handler)



"""
Save model checkpoint
"""
def save_checkpoint(model, model_path, epoch, train_loss, val_loss, all_epoch_train_losses,all_epoch_val_losses, all_epoch_train_accuracy_prec1, 
all_epoch_train_accuracy_prec5,all_epoch_val_accuracy_prec1, all_epoch_val_accuracy_prec5):
    if not os.path.exists(model_path):
        print("Checkpoint directory does not exist, {}".format(model_path))
        os.makedirs(model_path)
    else:
        print("Checkpoint exists")
    torch.save(model.state_dict(), model_path + '/ModelEpoch_{}_Train_loss_{:.4f}_Val_loss_{:.4f}.pth'.format(epoch, train_loss,val_loss))
    
    #pdb.set_trace()
    scipy.io.savemat(model_path + '/Losses_epoch_{}'.format(epoch), 
    {'train_loss': all_epoch_train_losses, 'val_loss':all_epoch_val_losses,
    'train_accuracy_prec1':all_epoch_train_accuracy_prec1, 'train_accuracy_prec5':all_epoch_train_accuracy_prec5,
    'val_accuracy_prec1':all_epoch_val_accuracy_prec1, 'val_accuracy_prec5':all_epoch_val_accuracy_prec5})

"""
Load Last saved model
"""
def load_last_model(model, model_path):
    models = glob(model_path + "/*.pth")
    if models:
        pdb.set_trace()
        model_ids = [(int(f.split('_')[-7]), f) for f in models]
        start_epoch, last_cp 	= max(model_ids, key=lambda item:item[0])
        print('Last checkpoint: ', last_cp)
        model.load_state_dict(torch.load(last_cp))
        all_metrics 		= scipy.io.loadmat(model_path + '/Losses_epoch_{}'.format(start_epoch))		
        all_epoch_train_losses 	= all_metrics['train_loss'][0]
        all_epoch_val_losses = all_metrics['val_loss'][0]
        all_epoch_train_accuracy_prec1 = all_metrics['train_accuracy_prec1'][0]
        all_epoch_train_accuracy_prec5 = all_metrics['train_accuracy_prec5'][0]
        all_epoch_val_accuracy_prec1 = all_metrics['val_accuracy_prec1'][0]
        all_epoch_val_accuracy_prec5 = all_metrics['val_accuracy_prec5'][0]
    else:
        start_epoch = 0
        last_cp = ''
        all_epoch_train_losses = np.array([])
        all_epoch_val_losses = np.array([])
        all_epoch_train_accuracy_prec1 = np.array([])
        all_epoch_train_accuracy_prec5 = np.array([])
        all_epoch_val_accuracy_prec1 = np.array([])
        all_epoch_val_accuracy_prec5 = np.array([])
    
    return start_epoch, all_epoch_train_losses.tolist(), all_epoch_val_losses.tolist(),all_epoch_train_accuracy_prec1.tolist(), all_epoch_train_accuracy_prec5.tolist(),all_epoch_val_accuracy_prec1.tolist(), all_epoch_val_accuracy_prec5.tolist()

    

def visualize_save_plots(train_loss, val_loss, train_acc, val_acc, confusion_matrixes):
    labels = ('Beco', 'Connie', 'Hank', 'Jati',
           'MyThai', 'Pheobe', 'Rudy', 'Sabu', 'Schottzie', 'Sunny')

    ## loss plot
    ## accuracy plot
    ## confusion matrix, heatamap
    cm = confusion_matrixes[np.argmax(val_acc)]
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%' % p
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    

    figsize =(20,10)
    fig = plt.figure(figsize=figsize)
    plt.subplot(2, 2, 1)
    plt.plot(train_acc, 'g')
    plt.plot(val_acc, 'g')
    plt.grid(True)
    plt.title("Train and Val Accuracy")

    plt.subplot(2, 2, 2)
    plt.plot(train_loss, 'g')
    plt.plot(val_loss, 'g')
    plt.grid(True)
    plt.title("Train and Val Loss")

    plt.subplot(2, 2, 3)
    sns.heatmap(cm, annot=annot, fmt='', cmap="Blues")
    plt.title("Val accuracy - actual vs predicted")
    plt.savefig("all_plots.png")


def visualize_test_acc(test_accs,confusion_mtxes,all_epochs):
    labels = ('Beco', 'Connie', 'Hank', 'Jati',
           'MyThai', 'Pheobe', 'Rudy', 'Sabu', 'Schottzie', 'Sunny')

    cm = confusion_matrix
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%' % p
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'


    figsize =(20,8)
    fig = plt.figure(figsize=figsize)
    
    plt.subplot(1, 2, 1)
    plt.plot(all_epochs,test_accs, 'g')
    plt.grid(True)
    plt.title("Test accuracy")
    
    plt.subplot(1, 2, 2)
    sns.heatmap(cm, annot=annot, fmt='', cmap="Blues")
    plt.title("Test accuracy - actual vs predicted")
   
    plt.savefig("classification_evaluation.png")
