from __future__ import print_function
import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from util.data_util import PartNormalDataset
import torch.nn.functional as F
import torch.nn as nn
import model as models
import numpy as np
from torch.utils.data import DataLoader
from util.util import to_categorical, compute_overall_iou, IOStream
from tqdm import tqdm
from collections import defaultdict
from torch.autograd import Variable
import random


classes_str = ['aero','bag','cap','car','chair','ear','guitar','knife','lamp','lapt','moto','mug','Pistol','rock','stake','table']

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='3D Shape Part Segmentation')
    parser.add_argument('--model', type=str, default='PointMLP1')
    parser.add_argument('--exp_name', type=str, default='demo1', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
                        help='number of episode to train')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--scheduler', type=str, default='step',
                        help='lr scheduler')
    parser.add_argument('--step', type=int, default=40,
                        help='lr decay step')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--manual_seed', type=int, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume training or not')
    parser.add_argument('--model_type', type=str, default='insiou',
                        help='choose to test the best insiou/clsiou/acc model (options: insiou, clsiou, acc)')
    return parser.parse_args()



def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)


def train(args, io):
    # ============= Model ===================
    num_part = 50
    device = torch.device("cuda" if args.cuda else "cpu")

    model = models.__dict__[args.model](num_part).to(device)
    io.cprint(str(model))

    model = nn.DataParallel(model)
    print("Let's use ", torch.cuda.device_count(), " GPUs!")

    best_acc = 0
    best_class_iou = 0
    best_instance_iou = 0
    optimizer_dict = None

    '''Resume or not'''
    start_epoch = 0 # start from epoch 0 or last checkpoint epoch
    if args.resume:
        checkpoint = torch.load("checkpoints/%s/last_model.pth" % args.exp_name, map_location=torch.device('cpu'))
        state_dict = checkpoint['model']
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        best_class_iou = checkpoint['best_class_iou']
        best_instance_iou = checkpoint['best_instance_iou']
        optimizer_dict = checkpoint['optimizer']
        for k in state_dict.keys():
            if 'module' not in k:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k in state_dict:
                    new_state_dict['module.' + k] = state_dict[k]
                state_dict = new_state_dict
            break
        model.load_state_dict(state_dict)

        print(f"Resume training model from epoch {start_epoch}...")
        print(torch.load("checkpoints/%s/last_model.pth" % args.exp_name).keys())
    else:
        print("Training from scratch...")

    # =========== Dataloader =================
    train_data = PartNormalDataset(npoints=2048, split='trainval', normalize=False)
    print("The number of training data is:%d", len(train_data))
    test_data = PartNormalDataset(npoints=2048, split='test', normalize=False)
    print("The number of test data is:%d", len(test_data))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=8, drop_last=False)
    # ============= Optimizer ================
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    if optimizer_dict is not None:
        opt.load_state_dict(optimizer_dict)
    if args.scheduler == 'cos':
        print("Use CosLR")
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr if args.use_sgd else args.lr /100,
                                      last_epoch=start_epoch - 1)
    else:
        print("Use StepLR")
        scheduler = StepLR(opt, step_size=args.step, gamma=0.5, last_epoch=start_epoch - 1)

    # ============= Training =================
    num_part = 50
    num_classes = 16

    for epoch in range(start_epoch, args.epochs):
        io.cprint(f"\n\n===> Epoch {str(epoch+1)}")
        train_epoch(train_loader, model, opt, scheduler, epoch, num_part, num_classes, io)
        test_metrics, total_per_cat_iou = test_epoch(test_loader, model, epoch, num_part, num_classes, io)

        # 1. when get the best accuracy, save the model:
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            io.cprint('saving new max Acc:%.5f' % best_acc)
            state = {
                'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'optimizer': opt.state_dict(), 'epoch': epoch, 'test_acc': best_acc}
            torch.save(state, 'checkpoints/%s/best_acc_model.pth' % args.exp_name)

        # 2. when get the best instance_iou, save the model:
        if test_metrics['shape_avg_iou'] > best_instance_iou:
            best_instance_iou = test_metrics['shape_avg_iou']
            io.cprint('saving new max instance iou:%.5f' % best_instance_iou)
            state = {
                'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'optimizer': opt.state_dict(), 'epoch': epoch, 'test_instance_iou': best_instance_iou}
            torch.save(state, 'checkpoints/%s/best_insiou_model.pth' % args.exp_name)

        # 3. when get the best class_iou, save the model:
        # first we need to calculate the average per-class iou
        class_iou = 0
        for cat_idx in range(16):
            class_iou += total_per_cat_iou[cat_idx]
        avg_class_iou = class_iou / 16
        if avg_class_iou > best_class_iou:
            best_class_iou = avg_class_iou
            # print the iou of each class:
            io.cprint('saving new max class iou:%.5f' % best_class_iou)
            for cat_idx in range(16):
                io.cprint(classes_str[cat_idx] + ' iou: ' + str(total_per_cat_iou[cat_idx]))
            state = {
                'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'optimizer': opt.state_dict(), 'epoch': epoch, 'test_class_iou': best_class_iou}
            torch.save(state, 'checkpoints/%s/best_clsiou_model.pth' % args.exp_name)

        # save last model
        state = {
            'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
            'optimizer': opt.state_dict(),
            'epoch': epoch,
            'best_acc':best_acc,
            'best_class_iou':best_class_iou,
            'best_instance_iou': best_instance_iou
        }
        torch.save(state, 'checkpoints/%s/last_model.pth' % (args.exp_name))

        # clear cache
        torch.cuda.empty_cache()

    # report best acc, ins_iou, cls_iou
    io.cprint('Final Max Acc:%.5f' % best_acc)
    io.cprint('Final Max instance iou:%.5f' % best_instance_iou)
    io.cprint('Final Max class iou:%.5f' % best_class_iou)



def train_epoch(train_loader, model, opt, scheduler, epoch, num_part, num_classes, io):
    train_loss = 0.0
    count = 0.0
    accuracy = []
    shape_ious = 0.0
    metrics = defaultdict(lambda: list())
    model.train()

    for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
        batch_size, num_point, _ = points.size()
        points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), \
                                          Variable(target.long()), Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(1).cuda(non_blocking=True), \
                                          target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
        # target: b,n
        seg_pred = model(points, norm_plt, to_categorical(label, num_classes))  # seg_pred: b,n,50
        loss = F.nll_loss(seg_pred.contiguous().view(-1, num_part), target.view(-1, 1)[:, 0])

        # instance iou without considering the class average at each batch_size:
        batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # list of of current batch_iou:[iou1,iou2,...,iou#b_size]
        # total iou of current batch in each process:
        batch_shapeious = seg_pred.new_tensor([np.sum(batch_shapeious)], dtype=torch.float64)  # same device with seg_pred!!!

        # Loss backward
        loss = torch.mean(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # accuracy
        seg_pred = seg_pred.contiguous().view(-1, num_part)  # b*n,50
        target = target.view(-1, 1)[:, 0]   # b*n
        pred_choice = seg_pred.contiguous().data.max(1)[1]  # b*n
        correct = pred_choice.eq(target.contiguous().data).sum()  # torch.int64: total number of correct-predict pts

        # sum
        shape_ious += batch_shapeious.item()  # count the sum of ious in each iteration
        count += batch_size   # count the total number of samples in each iteration
        train_loss += loss.item() * batch_size
        accuracy.append(correct.item()/(batch_size * num_point))   # append the accuracy of each iteration
        # Note: We do not need to calculate per_class iou during training


    scheduler.step()
    io.cprint('Learning rate: %f' % opt.param_groups[0]['lr'])

    metrics['accuracy'] = np.mean(accuracy)
    metrics['shape_avg_iou'] = shape_ious * 1.0 / count

    outstr = 'Train epoch %d, lr: %.6f loss: %.5f, train acc: %.3f, train ins_iou: %.3f' % \
             (epoch+1, opt.param_groups[0]['lr'], train_loss / count, metrics['accuracy'], metrics['shape_avg_iou'])
    io.cprint(outstr)


def test_epoch(test_loader, model, epoch, num_part, num_classes, io):
    test_loss = 0.0
    count = 0.0
    accuracy = []
    shape_ious = 0.0
    final_total_per_cat_iou = np.zeros(16).astype(np.float32)
    final_total_per_cat_seen = np.zeros(16).astype(np.int32)
    metrics = defaultdict(lambda: list())
    model.eval()

    # label_size: b, means each sample has one corresponding class
    for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
        with torch.no_grad():
            batch_size, num_point, _ = points.size()
            points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), \
                                              Variable(target.long()), Variable(norm_plt.float())
            points = points.transpose(2, 1)
            norm_plt = norm_plt.transpose(2, 1)
            points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(1).cuda(non_blocking=True), \
                                              target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
            seg_pred = model(points, norm_plt, to_categorical(label, num_classes))  # b,n,50

            # instance iou without considering the class average at each batch_size:
            batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # [b]
            # per category iou at each batch_size:

            for shape_idx in range(seg_pred.size(0)):  # sample_idx
                cur_gt_label = label[shape_idx]  # label[sample_idx], denotes current sample belongs to which cat
                final_total_per_cat_iou[cur_gt_label] += batch_shapeious[shape_idx]  # add the iou belongs to this cat
                final_total_per_cat_seen[cur_gt_label] += 1  # count the number of this cat is chosen

            # total iou of current batch in each process:
            batch_ious = seg_pred.new_tensor([np.sum(batch_shapeious)], dtype=torch.float64)  # same device with seg_pred!!!

            # prepare seg_pred and target for later calculating loss and acc:
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            # Loss
            loss = F.nll_loss(seg_pred.contiguous(), target.contiguous())

            # accuracy:
            pred_choice = seg_pred.data.max(1)[1]  # b*n
            correct = pred_choice.eq(target.data).sum()  # torch.int64: total number of correct-predict pts

            loss = torch.mean(loss)
            shape_ious += batch_ious.item()  # count the sum of ious in each iteration
            count += batch_size  # count the total number of samples in each iteration
            test_loss += loss.item() * batch_size
            accuracy.append(correct.item() / (batch_size * num_point))  # append the accuracy of each iteration

    for cat_idx in range(16):
        if final_total_per_cat_seen[cat_idx] > 0:  # indicating this cat is included during previous iou appending
            final_total_per_cat_iou[cat_idx] = final_total_per_cat_iou[cat_idx] / final_total_per_cat_seen[cat_idx]  # avg class iou across all samples

    metrics['accuracy'] = np.mean(accuracy)
    metrics['shape_avg_iou'] = shape_ious * 1.0 / count

    outstr = 'Test epoch %d, loss: %f, test acc: %f  test ins_iou: %f \n' % (epoch + 1, test_loss / count,
                                                                    metrics['accuracy'], metrics['shape_avg_iou'])

    io.cprint(outstr)

    return metrics, final_total_per_cat_iou


def test(args, io):
    # Dataloader
    test_data = PartNormalDataset(npoints=2048, split='test', normalize=False)
    print("The number of test data is:%d", len(test_data))

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=6,
                             drop_last=False)

    # Try to load models
    num_part = 50
    device = torch.device("cuda" if args.cuda else "cpu")

    model = models.__dict__[args.model](num_part).to(device)
    io.cprint(str(model))

    from collections import OrderedDict
    state_dict = torch.load("checkpoints/%s/best_%s_model.pth" % (args.exp_name, args.model_type),
                            map_location=torch.device('cpu'))['model']

    new_state_dict = OrderedDict()
    for layer in state_dict:
        new_state_dict[layer.replace('module.', '')] = state_dict[layer]
    model.load_state_dict(new_state_dict)

    model.eval()
    num_part = 50
    num_classes = 16
    metrics = defaultdict(lambda: list())
    hist_acc = []
    shape_ious = []
    total_per_cat_iou = np.zeros((16)).astype(np.float32)
    total_per_cat_seen = np.zeros((16)).astype(np.int32)

    for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
        batch_size, num_point, _ = points.size()
        points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(target.long()), Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze().cuda(
            non_blocking=True), target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)

        with torch.no_grad():
            seg_pred = model(points, norm_plt, to_categorical(label, num_classes))  # b,n,50

        # instance iou without considering the class average at each batch_size:
        batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # [b]
        shape_ious += batch_shapeious  # iou +=, equals to .append

        # per category iou at each batch_size:
        for shape_idx in range(seg_pred.size(0)):  # sample_idx
            cur_gt_label = label[shape_idx]  # label[sample_idx]
            total_per_cat_iou[cur_gt_label] += batch_shapeious[shape_idx]
            total_per_cat_seen[cur_gt_label] += 1

        # accuracy:
        seg_pred = seg_pred.contiguous().view(-1, num_part)
        target = target.view(-1, 1)[:, 0]
        pred_choice = seg_pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        metrics['accuracy'].append(correct.item() / (batch_size * num_point))

    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(hist_acc)
    metrics['shape_avg_iou'] = np.mean(shape_ious)
    for cat_idx in range(16):
        if total_per_cat_seen[cat_idx] > 0:
            total_per_cat_iou[cat_idx] = total_per_cat_iou[cat_idx] / total_per_cat_seen[cat_idx]

    # First we need to calculate the iou of each class and the avg class iou:
    class_iou = 0
    for cat_idx in range(16):
        class_iou += total_per_cat_iou[cat_idx]
        io.cprint(classes_str[cat_idx] + ' iou: ' + str(total_per_cat_iou[cat_idx]))  # print the iou of each class
    avg_class_iou = class_iou / 16
    outstr = 'Test :: test acc: %f  test class mIOU: %f, test instance mIOU: %f' % (metrics['accuracy'], avg_class_iou, metrics['shape_avg_iou'])
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    args = parse_args()
    args.exp_name = args.model+"_"+args.exp_name
    _init_()
    temp = "train" if not args.eval else "test"
    io = IOStream(f'checkpoints/{args.exp_name}/{temp}.log')
    io.cprint(str(args))

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        io.cprint('Using GPU')
        if args.manual_seed is not None:
            torch.cuda.manual_seed(args.manual_seed)
            torch.cuda.manual_seed_all(args.manual_seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)

