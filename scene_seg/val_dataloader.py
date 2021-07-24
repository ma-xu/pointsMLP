"""
Usage:
python train.py --model MLP --msg demo
"""

import os
import argparse
import time
import random
import numpy as np
import subprocess
import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler

from util import dataset, transform
from util.s3dis import S3DIS
from util.util import AverageMeter, intersectionAndUnionGPU, get_logger, get_parser
from utils import Logger, mkdir_p, progress_bar, save_model, save_args, cal_loss, get_screen_logger, set_seed
import models as models

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--num_classes', type=int, default=13, help='class_number')
    parser.add_argument('--model', default='MLP', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--num_point', type=int, default=4096, help='Point Number')
    parser.add_argument('--ignore_label', type=int, default=255, help='Point Number')

    parser.add_argument('--train_full_folder', type=str, default='dataset/s3dis/trainval_fullarea')
    parser.add_argument('--test_area', type=int, default=5)
    parser.add_argument('--block_size', type=float, default=1.0)
    parser.add_argument('--sample_rate', type=float, default=1.0)
    parser.add_argument('--fea_dim', type=int, default=6)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--train_batch_size_val', type=int, default=8)
    parser.add_argument('--data_root', default='dataset/s3dis')

    parser.add_argument('--val_list', type=str, default='dataset/s3dis/list/val5.txt')

    parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
    parser.add_argument('--learning_rate', default=0.05, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='decay rate')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--workers', default=8, type=int, help='workers')
    parser.add_argument('--no_transformation', action='store_true', default=False, help='do not use trnsformations')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Calculate mIoU during training, slow down significantly.')
    return parser.parse_args()


def prepare_data():
    if args.no_transformation:
        train_transform = None
    else:
        train_transform = transform.Compose([
            transform.RandomRotate(along_z=True),
            transform.RandomScale(scale_low=0.8, scale_high=1.2),
            transform.RandomJitter(sigma=0.01, clip=0.05),
            transform.RandomDropColor(color_augment=0.0)
        ])
    train_data = S3DIS(split='train', data_root=args.train_full_folder, num_point=args.num_point,
                       test_area=args.test_area, block_size=args.block_size, sample_rate=args.sample_rate,
                       transform=train_transform, fea_dim=args.fea_dim, shuffle_idx=False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)
    val_transform = transform.Compose([transform.ToTensor()])

    val_data = dataset.PointData(split='val', data_root=args.data_root, data_list=args.val_list,
                                 transform=val_transform, norm_as_feat=True, fea_dim=args.fea_dim)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.train_batch_size_val, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader


def main():
    global args
    args = parse_args()
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    train_loader, val_loader = prepare_data()

    for epoch in range(start_epoch, args.epoch):
        print(f"[epoch: {epoch}] Looping trainloader ....")
        time_cost = datetime.datetime.now()
        for i, (input, target) in enumerate(train_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            if i > 0 and i% args.print_freq == 0:
                time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
                print(f"Running {args.print_freq} iterations costs {time_cost}s [{input.shape}, {target,shape}]")
                time_cost = datetime.datetime.now()

        print(f"[epoch: {epoch}] Looping valloader ....")
        time_cost = datetime.datetime.now()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            if i > 0 and i% args.print_freq == 0:
                time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
                print(f"Running {args.print_freq} iterations costs {time_cost}s [{input.shape}, {target,shape}]")
                time_cost = datetime.datetime.now()






if __name__ == '__main__':
    main()
