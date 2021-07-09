"""
Plot the relationship (mean of heads) for global Transformer.
Plot the relationship of local-transformer and global-transformer.
python3 plot_relation2.py --id 26 --point_id 10 --stage 0 --save
"""
import argparse
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random
import os
import numpy as np
from collections import OrderedDict
import h5py
import math
import sys
sys.path.append("..")
from data import ModelNet40
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
# import models as models


import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
import torch.backends.cudnn as cudnn

from pathlib import Path
from tqdm import tqdm
from ShapeNetDataLoader import PartNormalDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda(non_blocking=True)
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_part_seg', help='model name')
    parser.add_argument('--batch_size', type=int, default=32, help='batch Size during training')
    parser.add_argument('--epoch', default=251, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')

    # for ploting
    parser.add_argument('--id', type=int, help='ID of the example')
    parser.add_argument('--stage', type=int, default=0, help='index of stage')
    parser.add_argument('--point_id', type=int, default=3, help='index of selected point in FPS')
    parser.add_argument('--head_id', type=int, default=0, help='index of selected head')
    parser.add_argument('--save', action='store_true', default=False, help='use normals besides x,y,z')
    parser.add_argument('--show', action='store_true', default=True, help='use normals besides x,y,z')
    parser.add_argument('--goon', action='store_true', default=False, help='use normals besides x,y,z')

    return parser.parse_args()

args = parse_args()
if args.id is None:
    args.id = np.random.randint(0,2874)


import matplotlib.colors as mcolors
def_colors = mcolors.CSS4_COLORS
colrs_list = []
for k,v in def_colors.items():
    colrs_list.append(k)
np.random.shuffle(colrs_list)

def plot_xyz(xyz, target, name="figures/figure.pdf"):
    fig = pyplot.figure()
    ax = Axes3D(fig)
    # ax = fig.gca(projection='3d')
    x_vals = xyz[:, 0]
    y_vals = xyz[:, 1]
    z_vals = xyz[:, 2]

    ax.set_xlim3d(min(x_vals)*0.9, max(x_vals)*0.9)
    ax.set_ylim3d(min(y_vals)*0.9, max(y_vals)*0.9)
    ax.set_zlim3d(min(z_vals)*0.9, max(z_vals)*0.9)
    for i in range(0,2048):
        col = int(target[i])
        ax.scatter(x_vals[i], y_vals[i], z_vals[i], c=colrs_list[col])

    ax.set_axis_off()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    # pyplot.tight_layout()
    if args.save:
        fig.savefig(name, bbox_inches='tight', pad_inches=0.00, transparent=True)

    pyplot.close()


def plot_xyz_color(xyz, name="figure.pdf", color=None ): # xyz: [n,3] selected_xyz:[3]
    fig = pyplot.figure()
    ax = Axes3D(fig)
    # ax = fig.gca(projection='3d')
    x_vals = xyz[:, 0]
    y_vals = xyz[:, 1]
    z_vals = xyz[:, 2]

    ax.set_xlim3d(min(x_vals)*0.9, max(x_vals)*0.9)
    ax.set_ylim3d(min(y_vals)*0.9, max(y_vals)*0.9)
    ax.set_zlim3d(min(z_vals)*0.9, max(z_vals)*0.9)


    if color is None:
        color = x_vals + y_vals + z_vals
        norm = pyplot.Normalize(vmin=min(color), vmax=max(color))
        ax.scatter(x_vals, y_vals, z_vals, c=color, cmap='hsv', norm=norm)
    else:
        ax.scatter(x_vals, y_vals, z_vals, c=color)

    ax.set_axis_off()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    # pyplot.tight_layout()

    fig.savefig(name, bbox_inches='tight', pad_inches=0.00, transparent=True)

    pyplot.close()


def main():


    # print(f"args: {args}")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    num_classes = 16
    num_part = 50

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)


    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    classifier = torch.nn.DataParallel(classifier)
    cudnn.benchmark = True
    exp_dir = Path('./checkpoints/')
    exp_dir = exp_dir.joinpath(args.log_dir)

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoints has been loaded.")
    except:
        print("Checkpoint path error!!!")
        return 0

    classifier.eval()
    root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test',
                                     normal_channel=args.normal)

    points, label, target = TEST_DATASET.__getitem__(args.id)
    xyz = points
    plot_xyz(xyz, target, name=f"figures/{args.id}-target.pdf")

    points = torch.tensor(points).unsqueeze(dim=0)
    label = torch.tensor(label).unsqueeze(dim=0)
    target = torch.tensor(target).unsqueeze(dim=0)


    points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
    print(f"Points shape: {points.shape} | label shape: {label.shape} | target shape: {target.shape}")
    points = points.transpose(2, 1)
    with torch.no_grad():
        predict, _ = classifier(points, to_categorical(label, num_classes))
        predict = predict.max(dim=-1)[1]
        predict = predict.squeeze(dim=0)
    print(f"Output shape: {predict.shape}")
    predict = predict.cpu().data.numpy()
    plot_xyz(xyz, predict, name=f"figures/{args.id}-predict.pdf")

    if args.goon:
        TEST_DATASET = PartNormalDataset(root=root, npoints=2048, split='test',
                                         normal_channel=args.normal)
        points, label, target = TEST_DATASET.__getitem__(args.id)
        plot_xyz_color(points, name=f"figures/{args.id}-input.pdf", color="mediumseagreen")

        TEST_DATASET = PartNormalDataset(root=root, npoints=2048, split='test',
                                         normal_channel=args.normal)
        points, label, target = TEST_DATASET.__getitem__(args.id)
        plot_xyz_color(points, name=f"figures/{args.id}-input2048.pdf")

        TEST_DATASET = PartNormalDataset(root=root, npoints=1024, split='test',
                                         normal_channel=args.normal)
        points, label, target = TEST_DATASET.__getitem__(args.id)
        plot_xyz_color(points, name=f"figures/{args.id}-input1024.pdf")

        TEST_DATASET = PartNormalDataset(root=root, npoints=512, split='test',
                                         normal_channel=args.normal)
        points, label, target = TEST_DATASET.__getitem__(args.id)
        plot_xyz_color(points, name=f"figures/{args.id}-input512.pdf")

        TEST_DATASET = PartNormalDataset(root=root, npoints=256, split='test',
                                         normal_channel=args.normal)
        points, label, target = TEST_DATASET.__getitem__(args.id)
        plot_xyz_color(points, name=f"figures/{args.id}-input256.pdf")

        TEST_DATASET = PartNormalDataset(root=root, npoints=128, split='test',
                                         normal_channel=args.normal)
        points, label, target = TEST_DATASET.__getitem__(args.id)
        plot_xyz_color(points, name=f"figures/{args.id}-input128.pdf")


if __name__ == '__main__':
    main()
