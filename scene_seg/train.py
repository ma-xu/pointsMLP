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
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--num_classes', type=int, default=13, help='class_number')
    parser.add_argument('--model', default='MLP', help='model name [default: pointnet_cls]')
    parser.add_argument('--epochs', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--num_point', type=int, default=4096, help='Point Number')
    parser.add_argument('--ignore_label', type=int, default=255, help='Point Number')

    parser.add_argument('--train_full_folder', type=str, default='dataset/s3dis/trainval_fullarea')
    parser.add_argument('--test_area', type=int, default=5)
    parser.add_argument('--block_size', type=float, default=1.0)
    parser.add_argument('--sample_rate', type=float, default=1.0)
    parser.add_argument('--fea_dim', type=int, default=6)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--train_batch_size_val', type=int, default=8)
    parser.add_argument('--data_root', default='dataset/s3dis')

    parser.add_argument('--val_list', type=str, default='dataset/s3dis/list/val5.txt')

    parser.add_argument('--print_freq', default=500, type=int, help='print frequency')
    parser.add_argument('--learning_rate', default=0.05, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='decay rate')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--workers', default=8, type=int, help='workers')
    parser.add_argument('--no_transformation', action='store_true', default=False, help='do not use trnsformations')
    return parser.parse_args()


def get_git_commit_id():
    try:
        cmd_out = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        return cmd_out
    except:
        # indicating no git found.
        return "0000000"

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
    global args, screen
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    message = time_str if args.msg is None else "-" + args.msg
    args.checkpoint = 'checkpoints/' + args.model + message
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    screen = get_screen_logger(os.path.join(args.checkpoint, 'screen.out'))
    screen.info(f"==> Start! Current commit version is: {get_git_commit_id()}")
    screen.info(f"==> args: {args}")

    screen.info("==> Building model..")
    net = models.__dict__[args.model](num_classes=args.num_classes)
    net = torch.nn.DataParallel(net.cuda())

    screen.info("===> Preparing criterion, optimizer, scheduler ...")
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=args.learning_rate / 100)

    best_mIoU = 0.0
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    if not os.path.isfile(os.path.join(args.checkpoint, "last_checkpoint.pth")):
        screen.info(f"==> Start training from scratch ...")
        save_args(args)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="S3DIS" + args.model)
        logger.set_names(["Epoch-Num", 'Learning-Rate',
                          'loss_train', 'mIoU_train', 'mAcc_train', 'allAcc_train',
                          'loss_val', 'mIoU_val', 'mAcc_val', 'allAcc_val'])
    else:
        screen.info(f"==> Resuming last checkpoint from {args.checkpoint} ")
        checkpoint_path = os.path.join(args.checkpoint, "last_checkpoint.pth")
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        best_mIoU = checkpoint['best_mIoU']
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="S3DIS" + args.model, resume=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    screen.info("===> preparing datasets ...")
    train_loader, val_loader = prepare_data()

    for epoch in range(start_epoch, args.epochs):
        screen.info('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, net, criterion, optimizer, epoch)
        loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, net, criterion)
        scheduler.step()
        is_best = False
        if mIoU_val > best_mIoU:
            best_mIoU = mIoU_val
            is_best = True
        save_model(
            net, epoch, path=args.checkpoint, is_best=is_best,
            best_mIoU=best_mIoU, commit_id=get_git_commit_id(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()
        )
        logger.append([epoch + 1, optimizer.param_groups[0]['lr'],
                       loss_train, mIoU_train, mAcc_train, allAcc_train,
                       loss_val, mIoU_val, mAcc_val, allAcc_val])

    screen.info(f"Done! best validation mIoU is {best_mIoU}")


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(input)
        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0:
            screen.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    screen.info(
        'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch + 1, args.epochs, mIoU,
                                                                                       mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    screen.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls
        output = model(input)
        loss = criterion(output, target)

        output = output.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0:
            screen.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    screen.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(args.classes):
        screen.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    screen.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    main()
