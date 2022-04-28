# This code is modified from: https://github.com/linusericsson/ssl-transfer/blob/main/finetune.py

#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from pprint import pprint
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms, models

import PIL
import numpy as np
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from datasets.custom_chexpert_dataset import CustomChexpertDataset
from datasets.custom_diabetic_retinopathy_dataset import CustomDiabeticRetinopathyDataset
from datasets.custom_stoic_dataset import CustomStoicDataset
from datasets.transforms import HistogramNormalize
from models.backbones import ResNetBackbone, ResNet18Backbone, DenseNetBackbone


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def count_acc(pred, label, metric):
    if metric == 'accuracy':
        return pred.eq(label.view_as(pred)).to(torch.float32).mean().item()
    elif metric == 'mean per-class accuracy':
        # get the confusion matrix
        cm = confusion_matrix(label.cpu(), pred.detach().cpu())
        cm = cm.diagonal() / cm.sum(axis=1)
        return cm.mean()
    elif metric == "auc":
        auc = roc_auc_score(label.cpu(),pred.detach().cpu())
        return auc



# Testing classes and functions

class FinetuneModel(nn.Module):
    def __init__(self, model, num_classes, steps, metric, device, feature_dim):
        super().__init__()
        self.num_classes = num_classes
        self.steps = steps
        self.metric = metric
        self.device = device
        self.model = nn.Sequential(model, nn.Linear(feature_dim, num_classes))
        self.model = self.model.to(self.device)
        self.model.train()
        self.criterion = nn.CrossEntropyLoss()

    def tune(self, train_loader, test_loader, lr, wd, early_stopping=False, val_loader=None, patience=3):
        # set up optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.steps)
        print(optimizer)
        logging.info(optimizer)
        # train the model with labels on the validation data
        self.model.train()
        train_loss = AverageMeter('loss', ':.4e')
        train_acc = AverageMeter('acc', ':6.2f')
        if early_stopping:
            best_acc = None
            early_stop_counter = 0
            early_stop = False
        step = 0
        pbar = tqdm(range(self.steps), desc='Training')
        running = True
        while running:
            for data, targets in train_loader:
                if step >= self.steps:
                    running = False
                    break

                targets = targets.type(torch.LongTensor)
                data, targets = data.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, targets)
                output = output.argmax(dim=1)
                # during training we can always track traditional accuracy, it'll be easier
                acc = 100. * count_acc(output, targets, "accuracy")
                loss.backward()

                optimizer.step()

                train_loss.update(loss.item(), data.size(0))
                train_acc.update(acc, data.size(0))
                pbar.update(1)
                pbar.set_postfix(loss=train_loss, acc=train_acc, lr=f"{scheduler.optimizer.param_groups[0]['lr']:.6f}")
                scheduler.step()

                step += 1

                # early stopping
                if early_stopping:
                    # check every 200 steps
                    if step % 200 == 0:
                        val_loss, val_acc = self.test_classifier(val_loader)
                        if best_acc is None:
                            best_acc = val_acc
                            best_state_dict = self.model.state_dict()
                        elif val_acc < best_acc:
                            early_stop_counter += 1
                            if early_stop_counter >= patience:
                                early_stop = True
                        else:
                            best_acc = val_acc
                            early_stop_counter = 0
                            best_state_dict = self.model.state_dict()

                    if early_stop:
                        print(f'Early stopping at step # {step} / 5000, best acc on val set {best_acc:.2f}%')
                        logging.info(f'Early stopping at step # {step} / 5000, best acc on val set {best_acc:.2f}%')
                        running = False
                        self.model.load_state_dict(best_state_dict)
                        break



        pbar.close()

        test_loss, test_acc = self.test_classifier(test_loader)
        return test_acc

    def test_classifier(self, data_loader):
        self.model.eval()
        test_loss, test_acc = 0, 0
        num_data_points = 0
        preds, labels = [], []
        with torch.no_grad():
            for i, (data, targets) in enumerate(tqdm(data_loader, desc=' Testing')):
                num_data_points += data.size(0)
                targets = targets.type(torch.LongTensor)
                data, targets = data.to(self.device), targets.to(self.device)
                output = self.model(data)
                tl = self.criterion(output, targets).item()
                tl *= data.size(0)
                test_loss += tl

                if self.metric == 'accuracy':
                    ta = 100. * count_acc(output.argmax(dim=1), targets, self.metric)
                    ta *= data.size(0)
                    test_acc += ta
                elif self.metric == 'mean per-class accuracy':
                    pred = output.argmax(dim=1).detach()
                    preds.append(pred)
                    labels.append(targets)
                elif self.metric == 'auc':
                    pred = output.argmax(dim=1).detach()
                    preds.append(pred)
                    labels.append(targets)


        if self.metric == 'accuracy':
            test_acc /= num_data_points
        elif self.metric == 'mean per-class accuracy':
            preds = torch.cat(preds)
            labels = torch.cat(labels)
            test_acc = 100. * count_acc(preds, labels, self.metric)
        elif self.metric == 'auc':
            preds = torch.cat(preds)
            labels = torch.cat(labels)
            test_acc = count_acc(preds, labels, self.metric)

        test_loss /= num_data_points

        self.model.train()
        return test_loss, test_acc


class FinetuneTester():
    def __init__(self, model_name, train_loader, val_loader, trainval_loader, test_loader,
                 metric, device, num_classes, feature_dim=2048, grid=None, steps=5000,
                 early_stopping=False, patience=3):
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.trainval_loader = trainval_loader
        self.test_loader = test_loader
        self.metric = metric
        self.device = device
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.grid = grid
        self.steps = steps
        self.early_stopping = early_stopping
        self.patience = patience
        self.best_params = {}

    def validate(self):
        best_score = 0
        for i, (lr, wd) in enumerate(self.grid):
            print(f'Run {i}')
            logging.info(f'Run {i}')
            print(f'lr={lr}, wd={wd}')
            logging.info(f'lr={lr}, wd={wd}')


            # load pretrained model
            if 'mimic-chexpert' in self.model_name:
                self.model = DenseNetBackbone(self.model_name)
                self.feature_dim = 1024
            elif 'mimic-cxr' in self.model_name:
                if 'r18' in self.model_name:
                    self.model = ResNet18Backbone(self.model_name)
                    self.feature_dim = 512
                else:
                    self.model = DenseNetBackbone(self.model_name)
                    self.feature_dim = 1024
            else:
                self.model = ResNetBackbone(self.model_name)
                self.feature_dim = 2048

            self.model = self.model.to(args.device)


            self.finetuner = FinetuneModel(self.model, self.num_classes, self.steps,
                                           self.metric, self.device, self.feature_dim)
            val_acc = self.finetuner.tune(self.train_loader, self.val_loader, lr, wd)
            print(f'Finetuned val accuracy {val_acc:.2f}%')
            logging.info(f'Finetuned val accuracy {val_acc:.2f}%')

            if val_acc > best_score:
                best_score = val_acc
                self.best_params['lr'] = lr
                self.best_params['wd'] = wd
                print(f"New best {self.best_params}")
                logging.info(f"New best {self.best_params}")

    def evaluate(self, lr=None, wd=None):
        if lr is not None:
            self.best_params['lr'] = lr
        if wd is not None:
            self.best_params['wd'] = wd

        print(f"Params {self.best_params}")
        logging.info(f"Params {self.best_params}")

        # load pretrained model
        if self.model_name in ['mimic-chexpert_lr_0.1', 'mimic-chexpert_lr_0.01', 'mimic-chexpert_lr_1.0', 'supervised_d121']:
            self.model = DenseNetBackbone(self.model_name)
            self.feature_dim = 1024
        elif 'mimic-cxr' in self.model_name:
            if 'r18' in self.model_name:
                self.model = ResNet18Backbone(self.model_name)
                self.feature_dim = 512
            else:
                self.model = DenseNetBackbone(self.model_name)
                self.feature_dim = 1024
        elif self.model_name == 'supervised_r18':
            self.model = ResNet18Backbone(self.model_name)
            self.feature_dim = 512
        else:
            self.model = ResNetBackbone(self.model_name)
            self.feature_dim = 2048

        self.model = self.model.to(args.device)


        self.finetuner = FinetuneModel(self.model, self.num_classes, self.steps,
                                       self.metric, self.device, self.feature_dim)
        if self.early_stopping:
            test_score = self.finetuner.tune(self.train_loader, self.test_loader,
                                             self.best_params['lr'], self.best_params['wd'],
                                             self.early_stopping, self.val_loader,
                                             self.patience)
        else:
            test_score = self.finetuner.tune(self.trainval_loader, self.test_loader,
                                             self.best_params['lr'], self.best_params['wd'],
                                             self.early_stopping)
        print(f'Finetuned test accuracy {test_score:.2f}%')
        logging.info(f'Finetuned test accuracy {test_score:.2f}%')
        return test_score

# Data classes and functions

def get_dataset(dset, root, split, transform): # Few shot is true! (used so labels are binary)
    return dset(root, train=(split == 'train'), transform=transform, download=True, few_shot = True)

def get_train_valid_loader(dset,
                           data_dir,
                           normalise_dict,
                           hist_norm,
                           batch_size,
                           image_size,
                           random_seed,
                           valid_size=0.2,
                           shuffle=True,
                           num_workers=1,
                           pin_memory=True,
                           data_augmentation=True):
    """
    Utility function for loading and returning train and valid
    multi-process iterators.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - dset: dataset class to load
    - normalise_dict: dictionary containing the normalisation parameters of the training set
    - batch_size: how many samples per batch to load.
    - image_size: size of images after transforms
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(**normalise_dict)


    # define transforms with augmentations
    if hist_norm:
        transform_aug = transforms.Compose([
            transforms.RandomResizedCrop(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            HistogramNormalize(),
        ])
    else:
        transform_aug = transforms.Compose([
            transforms.RandomResizedCrop(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])


    # define transform without augmentations
    if hist_norm:
        transform_no_aug = transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            HistogramNormalize(),
        ])
    else:
        transform_no_aug = transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])

    if not data_augmentation:
        transform_aug = transform_no_aug

    print("Train transform:", transform_aug)
    print("Val transform:", transform_no_aug)
    print("Trainval transform:", transform_aug)


    # select a random subset of the train set to form the validation set
    dataset = get_dataset(dset, data_dir, 'train', transform_aug)
    valid_dataset = get_dataset(dset, data_dir, 'train', transform_no_aug)

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    trainval_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader, trainval_loader


def get_test_loader(dset,
                    data_dir,
                    normalise_dict,
                    hist_norm,
                    batch_size,
                    image_size,
                    shuffle=False,
                    num_workers=1,
                    pin_memory=True):
    """
    Utility function for loading and returning a multi-process
    test iterator.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - dset: dataset class to load
    - normalise_dict: dictionary containing the normalisation parameters of the training set
    - batch_size: how many samples per batch to load.
    - image_size: size of images after transforms
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """

    normalize = transforms.Normalize(**normalise_dict)

    # define transform
    if hist_norm:
        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            HistogramNormalize(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])

    print("Test transform:", transform)

    dataset = get_dataset(dset, data_dir, 'test', transform)

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


def prepare_data(dset, data_dir, batch_size, image_size, normalisation, hist_norm, num_workers, data_augmentation):
    print(f'Loading {dset} from {data_dir}, with batch size={batch_size}, image size={image_size}, norm={normalisation}')
    logging.info(f'Loading {dset} from {data_dir}, with batch size={batch_size}, image size={image_size}, norm={normalisation}')
    if normalisation:
        normalise_dict = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    else:
        normalise_dict = {'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]}
    train_loader, val_loader, trainval_loader = get_train_valid_loader(dset, data_dir, normalise_dict, hist_norm,
                                                batch_size, image_size, random_seed=0, num_workers=num_workers,
                                                pin_memory=False, data_augmentation=data_augmentation)
    test_loader = get_test_loader(dset, data_dir, normalise_dict, hist_norm, batch_size, image_size, num_workers=num_workers,
                                                pin_memory=False)
    return train_loader, val_loader, trainval_loader, test_loader



# name: {class, root, num_classes, metric}
FINETUNE_DATASETS = {
    'chexpert': [CustomChexpertDataset, './data/chexpert', 2, 'auc'],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate pretrained self-supervised model via finetuning.')
    parser.add_argument('-m', '--model', type=str, default='moco-v2', help='name of the pretrained model to load and evaluate')
    parser.add_argument('-d', '--dataset', type=str, default='chexpert', help='name of the dataset to evaluate on')
    parser.add_argument('-b', '--batch-size', type=int, default=64, help='the size of the mini-batches when inferring features')
    parser.add_argument('-i', '--image-size', type=int, default=224, help='the size of the input images')
    parser.add_argument('-w', '--workers', type=int, default=4, help='the number of workers for loading the data')
    parser.add_argument('-s', '--search', action='store_true', default=False, help='whether to perform a hyperparameter search on the lr and wd')
    parser.add_argument('-g', '--grid-size', type=int, default=2, help='the number of learning rate values in the search grid')
    parser.add_argument('-e', '--early-stopping', action='store_true', default=False, help='whether to perform early stopping')
    parser.add_argument('-p', '--patience', type=int, default=3, help='patience in units of 200 steps for early stopping')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-8, help='weight decay')
    parser.add_argument('--steps', type=int, default=5000, help='the number of finetuning steps')
    parser.add_argument('--no-da', action='store_true', default=False, help='disables data augmentation during training')
    parser.add_argument('-n', '--no-norm', action='store_true', default=False,
                        help='whether to turn off data normalisation (based on ImageNet values)')
    parser.add_argument('--device', type=str, default='cuda', help='CUDA or CPU training (cuda | cpu)')
    args = parser.parse_args()
    args.norm = not args.no_norm
    args.da = not args.no_da
    del args.no_norm
    del args.no_da
    pprint(args)


    # histogram normalization
    hist_norm = False
    if 'mimic-chexpert' in args.model:
        hist_norm = True


    # set-up logging
    log_fname = f'{args.dataset}_auc.log'
    if not os.path.isdir(f'./logs/finetune_auc/{args.model}'):
        os.makedirs(f'./logs/finetune_auc/{args.model}')
    log_path = os.path.join(f'./logs/finetune_auc/{args.model}', log_fname)
    logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO)
    logging.info(args)


    # load dataset
    dset, data_dir, num_classes, metric = FINETUNE_DATASETS[args.dataset]
    train_loader, val_loader, trainval_loader, test_loader = prepare_data(
        dset, data_dir, args.batch_size, args.image_size, normalisation=args.norm,
        hist_norm=hist_norm, num_workers=args.workers, data_augmentation=args.da)

    # set up learning rate and weight decay ranges
    lr = torch.logspace(-4, -1, args.grid_size).flip(dims=(0,))
    wd = torch.cat([torch.zeros(1), torch.logspace(-6, -3, args.grid_size)])
    grid = [(l.item(), (w / l).item()) for l in lr for w in wd]


    # evaluate model on dataset by finetuning
    tester = FinetuneTester(args.model, train_loader, val_loader, trainval_loader, test_loader,
                            metric, args.device, num_classes, grid=grid, steps=args.steps,
                            early_stopping=args.early_stopping, patience=args.patience)

    if args.search:
        print('Performing hyperparameter search for lr and wd')
        # tune hyperparameters
        tester.validate()
        # use best hyperparameters to finally evaluate the model
        test_score = tester.evaluate()
    else:
        test_score = tester.evaluate(args.lr, args.wd)
