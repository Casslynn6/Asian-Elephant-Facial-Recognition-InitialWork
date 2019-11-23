import os
import shutil
import torch
import torchvision.transforms as transforms
# import torchvision.datasets as datasets
from data.folder_new import ImageFolder_new


def generate_dataloader(args):
    # Data loading code
    # the dataloader for the target dataset.
    traindir = os.path.join(args.data_path, 'splited_image/train')
    valdir = os.path.join(args.data_path, 'splited_image/val')
    if not os.path.isdir(traindir):
        split_train_test_images(args.data_path)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = ImageFolder_new(
        traindir,
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
        ImageFolder_new(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # the dataloader for the source dataset.
    if args.auxiliary_dataset == 'imagenet':
        traindir_source = os.path.join(args.data_path_source, 'Data/CLS-LOC/train')
        valdir_source = os.path.join(args.data_path_source, 'Data/CLS-LOC/val')
    else:
        #traindir_source = args.data_path_source
        traindir_source = os.path.join(args.data_path_source, 'L-Bird-Subset') ##L-Bird-Whole-Condensed
        valdir_source = os.path.join(args.data_path_source, 'L-Bird-Subset-val')
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
        return train_loader_source, val_loader_source, train_loader, val_loader
    else:
        return train_loader, val_loader