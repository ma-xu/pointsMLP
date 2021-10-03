"""
Plot the parts.
python plot_ptseg.py --model model31G --exp_name demo1 --id 1
"""
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

def test(args):
    # Dataloader
    test_data = PartNormalDataset(npoints=2048, split='test', normalize=False)
    print("===> The number of test data is:%d", len(test_data))
    # Try to load models
    print("===> Create model...")
    num_part = 50
    device = torch.device("cuda" if args.cuda else "cpu")
    model = models.__dict__[args.model](num_part).to(device)
    print("===> Load checkpoint...")
    from collections import OrderedDict
    state_dict = torch.load("checkpoints/%s/best_%s_model.pth" % (args.exp_name, args.model_type),
                            map_location=torch.device('cpu'))['model']
    new_state_dict = OrderedDict()
    for layer in state_dict:
        new_state_dict[layer.replace('module.', '')] = state_dict[layer]
    model.load_state_dict(new_state_dict)
    print("===> Start evaluate...")
    model.eval()
    num_classes = 16
    points, label, target, norm_plt = test_data.__getitem__(args.id)
    points = torch.tensor(points).unsqueeze(dim=0)
    label = torch.tensor(label).unsqueeze(dim=0)
    target = torch.tensor(target).unsqueeze(dim=0)
    norm_plt = torch.tensor(norm_plt).unsqueeze(dim=0)
    points = points.transpose(2, 1)
    norm_plt = norm_plt.transpose(2, 1)
    points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze().cuda(
        non_blocking=True), target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
    with torch.no_grad():
            cls_lable = to_categorical(label, num_classes)
            print(f"cls_lable.shape is {cls_lable.shape}")
            seg_pred = model(points, norm_plt, cls_lable)  # b,n,50

    print(f"label shape is: {label.shape}")
    print(f"seg_pred shape is: {seg_pred.shape}")


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='3D Shape Part Segmentation')
    parser.add_argument('--model', type=str, default='PointMLP1')
    parser.add_argument('--id', type=int, default='1')
    parser.add_argument('--exp_name', type=str, default='demo1', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--model_type', type=str, default='insiou',
                        help='choose to test the best insiou/clsiou/acc model (options: insiou, clsiou, acc)')

    args = parser.parse_args()
    args.exp_name = args.model+"_"+args.exp_name
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    test(args)
