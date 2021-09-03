"""
for training with resume functions.
Usage:
python main.py --model PointNet --msg demo
or
CUDA_VISIBLE_DEVICES=0 nohup python main.py --model PointNet --msg demo > nohup/PointNet_demo.out &
"""
import argparse
import os
import logging
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--path', default='./checkpoints', type=str)
    parser.add_argument('--model', type=str)
    return parser.parse_args()


def main():
        args = parse_args()
        models = os.listdir(args.path)
        for model in models:
            if args.model is not None and args.model not in model:
                continue
            checkpoint_path = os.path.join(args.path, model, "best_checkpoint.pth")
            if os.path.isfile(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                epoch = checkpoint['epoch']
                best_test_acc = checkpoint['best_test_acc']

                print(f"{model}: epoch{epoch}  best_test_acc: {best_test_acc}")






if __name__ == '__main__':
    main()
