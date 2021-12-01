"""
for training with resume functions.
Usage:
python main.py --model PointNet --msg demo
or
CUDA_VISIBLE_DEVICES=0 nohup python main.py --model PointNet --msg demo > nohup/PointNet_demo.out &
"""
import argparse
import os
import subprocess
import logging
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import models as models
from utils import Logger, mkdir_p, progress_bar, save_model, save_args, cal_loss
from ScanObjectNN import ScanObjectNN
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics
import numpy as np


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='PointNet', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_classes', default=15, type=int, help='default value for classes of ScanObjectNN')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--smoothing', action='store_true', default=False, help='loss smoothing')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--workers', default=4, type=int, help='workers')
    return parser.parse_args()

def get_git_commit_id():
    try:
        cmd_out = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        return cmd_out
    except:
        # indicating no git found.
        return "0000000"

def main():
    args = parse_args()
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    if args.seed is not None:
        torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = 'cuda'
        if args.seed is not None:
            torch.cuda.manual_seed(args.seed)
    else:
        device = 'cpu'
    # time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    # if args.msg is None:
    #     message = time_str
    # else:
    message = "-" + args.msg
    args.checkpoint = 'checkpoints/' + args.model + message
    # if not os.path.isdir(args.checkpoint):
    #     mkdir_p(args.checkpoint)

    # screen_logger = logging.getLogger("Model")
    # screen_logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(message)s')
    # file_handler = logging.FileHandler(os.path.join(args.checkpoint, "out.txt"))
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(formatter)
    # screen_logger.addHandler(file_handler)

    # def printf(str):
    #     screen_logger.info(str)
    #     print(str)


    print('==> Building model..')
    net = models.__dict__[args.model](num_classes=args.num_classes)
    criterion = cal_loss
    net = net.to(device)
    # criterion = criterion.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True



    print(f"Resuming last checkpoint from {args.checkpoint}")
    checkpoint_path = os.path.join(args.checkpoint, "best_checkpoint.pth")
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']
    best_test_acc = checkpoint['best_test_acc']
    best_train_acc = checkpoint['best_train_acc']
    best_test_acc_avg = checkpoint['best_test_acc_avg']
    best_train_acc_avg = checkpoint['best_train_acc_avg']
    best_test_loss = checkpoint['best_test_loss']
    best_train_loss = checkpoint['best_train_loss']
    # logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model, resume=True)
    # optimizer_dict = checkpoint['optimizer']

    print('==> Preparing data..')
    test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points), num_workers=args.workers,
                             batch_size=args.batch_size, shuffle=True, drop_last=False)


    test_out = validate(net, test_loader, criterion, device)

    print(f'saved loss: {best_test_loss},  test loss: {test_out["loss"]} ')
    print(f'saved acc_avg: {best_test_acc_avg},  test acc_avg: {test_out["acc_avg"]} ')
    print(f'saved acc: {best_test_acc},  test acc: {test_out["acc"]} ')



def validate(net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = net(data)
            loss = criterion(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }


if __name__ == '__main__':
    main()
