"""
Plot the parts.
python plot_ptseg.py --model model31G --exp_name demo1 --id 1
"""
from __future__ import print_function
import os
import argparse
import torch
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from torch.autograd import Variable
import random

# import matplotlib.colors as mcolors
# def_colors = mcolors.CSS4_COLORS
# colrs_list = []
# np.random.seed(2021)
# for k, v in def_colors.items():
#     colrs_list.append(k)
# np.random.shuffle(colrs_list)
colrs_list = [
    "C6", "C1","C2","C3","C4","C5","C6","C7","C8","C9",
    "deepskyblue", "tan","orangered","C9","tan","c","y", "gold","darkorange","g",
    "orangered","tomato","tan","darkorchid","C1","tomato", "y","C5","C3","C4",
    "C5","C6","C7","C8","C9","deepskyblue", "m","deeppink","hotpink","lime",
    "c","y", "gold","darkorange","g","orangered","tomato","tan","darkorchid","violet",
    "C0", "C1","C2","C3","C4","C5","C6","C7","C8","C9",
    "deepskyblue", "m","deeppink","hotpink","lime","c","y",
    "gold","darkorange","g","orangered","tomato","tan","darkorchid","violet"
]

def test(args):

    # np.savetxt(f"figures/{args.id}-point.txt", points)
    # np.savetxt(f"figures/{args.id}-target.txt", target)
    # np.savetxt(f"figures/{args.id}-predict.txt", predict)
    points = np.recfromtxt(f"{args.id}-point.txt")
    # print(points.shape)
    target = np.recfromtxt(f"{args.id}-target.txt")
    # target = target[:400]
    predict = np.recfromtxt(f"{args.id}-predict.txt")
    print(f"unique label is: {np.unique(predict)}")
    # predict = predict[:400]
    # start plot
    print(f"===> stat plotting")
    plot_xyz(points, target, name=f"{args.id}-gt.pdf")
    plot_xyz(points, predict, name=f"{args.id}-predict.pdf")


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
        ax.scatter(x_vals[i], y_vals[i], z_vals[i], c=colrs_list[col], marker=".", s=200, alpha=0.6)
    ax.set_axis_off()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    # ax.view_init(30,30)
    # pyplot.tight_layout()
    pyplot.show()
    fig.savefig(name, bbox_inches='tight', pad_inches=-0., transparent=True)
    pyplot.close()

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
