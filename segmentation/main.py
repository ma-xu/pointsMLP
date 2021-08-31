"""
Author: Benny
Date: Nov 2019
Usage:
python
"""
import argparse
import os
from S3DISDataLoader import S3DISDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import models as models
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--root', type=str, default='data/stanford_indoor3d/', help='data path')
    parser.add_argument('--model', type=str, default='model31G', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=100, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--scheduler', type=str, default='cos', help='cos or step]')
    parser.add_argument('--exp_name', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=30, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--workers', default=4, type=int, help='workers')
    return parser.parse_args()


def main(args):
    def printf(str):
        logger.info(str)
        print(str)

    '''CREATE DIR'''
    time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    folder_name = time_str if args.exp_name is None else args.exp_name
    folder_name = args.model + folder_name
    experiment_dir = Path('./checkpoints/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(folder_name)
    experiment_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/out.txt' % (experiment_dir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    printf('PARAMETER ...')
    printf(args)


    NUM_CLASSES = 13
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    printf("start loading training data ...")
    TRAIN_DATASET = S3DISDataset(split='train', data_root=args.root, num_point=NUM_POINT,
                                 test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    printf("The number of training data is: %d" % len(TRAIN_DATASET))
    printf("start loading test data ...")
    TEST_DATASET = S3DISDataset(split='test', data_root=args.root, num_point=NUM_POINT,
                                test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    printf("The number of test data is: %d" % len(TEST_DATASET))
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=args.workers,pin_memory=True, drop_last=True,
                        worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=args.workers, pin_memory=True, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    '''MODEL LOADING'''
    classifier = models.__dict__[args.model]()
    classifier = classifier.cuda()
    criterion = provider.get_loss().cuda()
    cudnn.benchmark = True


    best_iou = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    optimizer_dict = None

    if os.path.isfile(os.path.join(experiment_dir, "last_checkpoint.pth")):
        checkpoint = torch.load(str(experiment_dir) + 'last_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer_dict = checkpoint['optimizer']
        printf('resume pretrain model')
    else:
        printf('No existing model, starting training from scratch...')

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam( classifier.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate*100.0, momentum=0.9)
    if optimizer_dict is not None:
        optimizer.load_state_dict(optimizer_dict)
    if args.scheduler =="cos":
        eta_min = args.learning_rate/50.0 if args.optimizer == 'Adam' else args.learning_rate
        scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=eta_min, last_epoch=start_epoch - 1)
    else:
        scheduler = StepLR(optimizer, args.step_size, gamma=args.lr_decay)


    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        printf('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0.
        classifier = classifier.train()

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            points = points.data.numpy()
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1) # [b,d,n]

            seg_pred = classifier(points) # [b,n,num_classes]
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss.item()
        printf('Training mean loss: %.6f  Training accuracy: %.6f' %
               (loss_sum / num_batches, total_correct / float(total_seen)))
        scheduler.step()

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, weights)
                loss_sum += loss.item()
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
            printf('eval mean loss: %f, avg class IoU: %f, accuracy: %f, avg class acc: %f' %
                   (loss_sum / float(num_batches),
                    mIoU,
                    total_correct / float(total_seen),
                    np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))
                    ))
            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            printf(iou_per_class_str)

            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save new best model...')
                savepath = str(experiment_dir) + '/best_model.pth'
                state = {
                    'epoch': epoch,
                    "best_iou": best_iou,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                printf('Saving model....')
            printf('Best mIoU: %f' % best_iou)

            savepath = str(experiment_dir) + '/last_model.pth'
            state = {
                'epoch': epoch,
                "best_iou": best_iou,
                'class_avg_iou': mIoU,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)


if __name__ == '__main__':
    args = parse_args()
    main(args)
