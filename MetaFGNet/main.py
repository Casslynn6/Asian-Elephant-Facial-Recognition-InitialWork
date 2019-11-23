#!/usr/bin/env python
# coding: utf-8

# In[116]:


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
from models.densenet import densenet 

import torchvision
from trainer import train, validate
# Configuration Options
from utils import get_config


# In[117]:


torch.cuda.is_available()


# ## Read Opts

# In[118]:
best_prec1 = 0

from collections import namedtuple
d = get_config(config="configs/elephant.yaml")
d["log"] =""
args = namedtuple("args", d.keys())(*d.values())
log = args.log + '_' + args.arch + '_' + args.dataset + '_' + str(args.batch_size) + 'Timg_' + args.auxiliary_dataset                + '_' + str(args.batch_size_source) + 'Simg_Meta_train_Lr' + str(args.meta_train_lr) + '_' +               str(args.num_updates_for_gradient)
log


# ## Data Loaders

# In[50]:


import torchvision.transforms as transforms
from data.folder_new import ImageFolder_new

def generate_dataloader(target_train_path, target_val_path, aux_train_path, aux_val_path):
    
    # Data loading code
    # the dataloader for the target dataset.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_dataset = ImageFolder_new(
        target_train_path,
        transforms.Compose([
            # transforms.Resize(256),
            # transforms.RandomCrop(224),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers = args.workers, pin_memory=True, sampler=None
    )
    
    val_loader = torch.utils.data.DataLoader(
        ImageFolder_new(target_val_path, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    
    # the dataloader for the source dataset.
    traindir_source = aux_train_path
    valdir_source = aux_val_path
    
    if len(os.listdir(traindir_source)) != 0:
        normalize_source = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        train_dataset_source = ImageFolder_new(
            traindir_source,
            transforms.Compose([
                #transforms.Resize(256),
                #transforms.RandomCrop(224),
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_source,
            ])
        )
        train_loader_source = torch.utils.data.DataLoader(
            train_dataset_source, batch_size=args.batch_size_source, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None
        )

        val_loader_source = torch.utils.data.DataLoader(
            ImageFolder_new(valdir_source, transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize_source,
            ])),
            batch_size=args.batch_size_source, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        print(f"No of train source:{len(train_loader_source)}, No of source val: {len(val_loader_source)},No. of target train :{len(train_loader)}, No of target val: {len(val_loader)}")
        return train_loader_source, val_loader_source, train_loader, val_loader
    else:
        return train_loader, val_loader


# In[51]:


#generate_dataloader(os.path.join(args.data_path,"train"), os.path.join(args.data_path,"val"), os.path.join(args.auxiliary_dataset,"train"),os.path.join(args.auxiliary_dataset,"val"))


# ## Model

# In[52]:
def save_checkpoint(state, is_best, args, epoch):
    filename = str(epoch) + 'checkpoint.pth.tar'
    dir_save_file = os.path.join(args.log_folder, filename)
    torch.save(state, dir_save_file)
    if is_best:
        shutil.copyfile(dir_save_file, os.path.join(args.log_folder, 'model_best.pth.tar'))


model_source, model_target = resnet(args) if 'resnet' in args.arch else densenet(args)


# In[54]:


# define-multi GPU
model_source = torch.nn.DataParallel(model_source).cuda()
model_target = torch.nn.DataParallel(model_target).cuda()


# In[55]:


print('the memory id should be same for the shared feature extractor:')
print(id(model_source.module.resnet_conv))   # the memory is shared here
print(id(model_target.module.resnet_conv))


# In[56]:


print('the memory id should be different for the different classifiers:')
print(id(model_source.module.fc))  # the memory id shared here.
print(id(model_target.module.fc))


# ## Initialize 

# In[57]:


# define loss function(criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda()

np.random.seed(1)  ### fix the random data.
random.seed(1)


# In[62]:


## Use Meta-SGD to update parameteres
if args.meta_sgd:
    meta_train_lr = []
    for param in model_target.parameters():
        meta_train_lr.append(torch.FloatTensor(param.data.size()).fill_(args.meta_train_lr).cuda())


# In[66]:


args.pretrained ## Pretrained settings for Optimizer


# In[65]:


optimizer = torch.optim.SGD([
            {'params': model_source.module.resnet_conv.parameters(), 'name': 'new-added'},
            {'params': model_source.module.fc.parameters(), 'name': 'new-added'},
            {'params': model_target.module.fc.parameters(), 'name': 'new-added'},
        ],
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=float(args.weight_decay))


# ## Training

# In[67]:


## If Resume is true


if args.resume:
    if os.path.isfile(args.resume):
        print("==> loading checkpoints '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        if args.meta_sgd:
            meta_train_lr = checkpoint['meta_train_lr']
        best_prec1 = checkpoint['best_prec1']
        model_source.load_state_dict(checkpoint['source_state_dict'])
        model_target.load_state_dict(checkpoint['target_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("==> loaded checkpoint '{}'(epoch {})"
                  .format(args.resume, checkpoint['epoch']))
    else:
        raise ValueError('The file to be resumed from is not exited', args.resume)


# In[73]:


if not os.path.isdir(args.log_folder):
    os.makedirs(args.log_folder)
log = open(os.path.join(args.log_folder, 'log.txt'), 'w')
state = {k: v for k, v in d.items()}
log.write(json.dumps(state) + '\n')
log.close()


# In[74]:


cudnn.benchmark = True
dataloader_returned = generate_dataloader(os.path.join(args.data_path,"train"), os.path.join(args.data_path,"val"), os.path.join(args.auxiliary_dataset,"train"),os.path.join(args.auxiliary_dataset,"val"))


# In[75]:


dataloader_number_returned = len(dataloader_returned)


# In[79]:


print(f'the number of dataloader {dataloader_number_returned} returned   ' )


# In[82]:


if dataloader_number_returned != 2:
    train_loader_source, val_loader_source, train_loader_target, val_loader_target = dataloader_returned
else:
    train_loader_target, val_loader_target = dataloader_returned
    train_loader_source = None


# In[83]:


print('begin training')


# In[91]:


if train_loader_source:
    train_loader_source_batch = enumerate(train_loader_source)
else:
    train_loader_source_batch = None


# In[90]:


train_loader_target_batch = enumerate(train_loader_target)


# In[119]:


print('begin training')
if train_loader_source:
    train_loader_source_batch = enumerate(train_loader_source)
else:
    train_loader_source_batch = None
train_loader_target_batch = enumerate(train_loader_target)
for epoch in range(args.start_epoch, args.epochs):
    if args.meta_sgd:
        train_loader_source_batch, train_loader_target_batch, meta_train_lr = train(train_loader_source, train_loader_source_batch, train_loader_target,train_loader_target_batch, model_source, model_target, criterion, optimizer, epoch, args, meta_train_lr)
    else:
        train_loader_source_batch, train_loader_target_batch = train(train_loader_source, train_loader_source_batch, train_loader_target,train_loader_target_batch, model_source, model_target, criterion, optimizer, epoch, args, None)
    # train(train_loader, model, criterion, optimizer, epoch, args)
    # evaluate on the val data
    if (epoch + 1) % args.test_freq == 0 or (epoch + 1) % args.epochs == 0:
        if dataloader_number_returned == 2:
            prec1 = validate(None, val_loader_target, model_source, model_target, criterion, epoch, args)
        else:
            prec1 = validate(val_loader_source, val_loader_target, model_source, model_target, criterion, epoch, args)
        # prec1 = 1
        # record the best prec1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            log = open(os.path.join(args.log, 'log.txt'), 'a')
            log.write('     \nTarget_T1 acc: %3f' % (best_prec1))
            log.close()
        if args.meta_sgd:
            save_checkpoint({
                'epoch': epoch + 1,
                'meta_train_lr': meta_train_lr,
                'arch': args.arch,
                'source_state_dict': model_source.state_dict(),
                'target_state_dict': model_target.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args, epoch)
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'source_state_dict': model_source.state_dict(),
                'target_state_dict': model_target.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args, epoch + 1)







