"""
for training with resume functions.
Usage:
python main.py --model PointNet --msg demo
or
CUDA_VISIBLE_DEVICES=0 nohup python main.py --model PointNet --msg demo > nohup/PointNet_demo.out &
"""
import argparse
import os
import random

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utils import Logger, mkdir_p, progress_bar, save_model, save_args, cal_loss
from ScanObjectNN import ScanObjectNN

import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--num_points', type=int, default=5000, help='Point Number')
    parser.add_argument('--id', default=800, type=int, help='ID of the example 2468')
    parser.add_argument('--save', action='store_true', default=False, help='use normals besides x,y,z')
    return parser.parse_args()


def set_seed(seed=None):
    if seed is None:
        return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = ("%s" % seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def plot_xyz(xyz, args,  name="figure.pdf" ): # xyz: [n,3] selected_xyz:[3]
    fig = pyplot.figure()
    ax = Axes3D(fig)
    # ax = fig.gca(projection='3d')
    x_vals = xyz[:, 0]
    y_vals = xyz[:, 1]
    z_vals = xyz[:, 2]

    ax.set_xlim3d(min(x_vals)*0.9, max(x_vals)*0.9)
    ax.set_ylim3d(min(y_vals)*0.9, max(y_vals)*0.9)
    ax.set_zlim3d(min(z_vals)*0.9, max(z_vals)*0.9)

    color = x_vals+y_vals+z_vals
    norm = pyplot.Normalize(vmin=min(color), vmax=max(color))
    ax.scatter(x_vals, y_vals, z_vals, c=color, cmap='hsv', norm=norm)

    ax.set_axis_off()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    # pyplot.tight_layout()
    if args.show:
        pyplot.show()
    if args.save:
        fig.savefig(name, bbox_inches='tight', pad_inches=0.00, transparent=True)

    pyplot.close()

def main():
    args = parse_args()
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    if not os.path.isdir("figures"):
        mkdir_p("figures")

    print('==> Preparing data ...')
    # train_set =ModelNet40(partition='train', num_points=args.num_points)
    test_set = ScanObjectNN(partition='test', num_points=args.num_points)

    data, label = test_set.__getitem__(args.id)
    print('==> plotting ...')
    plot_xyz(data, args, name=f"figures/Image-{args.id}-{args.num_points}.pdf" )



if __name__ == '__main__':
    set_seed(32) # must
    main()



if __name__ == '__main__':
    main()
