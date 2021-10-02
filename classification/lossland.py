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
import models as models
from utils import Logger, mkdir_p, progress_bar, save_model, save_args, cal_loss
from data import ModelNet40
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics
from collections import OrderedDict
import numpy as np


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='PointNet', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--workers', default=8, type=int, help='workers')
    return parser.parse_args()


def load_pretrained(args):
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(os.path.join("ablation_checkpoints",args.model+'-loss','best_checkpoint.pth'),
                            map_location=torch.device('cpu'))
    new_dict = OrderedDict()
    for k, v in checkpoint['net'].items():
        # if k.startswith("module.1."):
        #     k = k[9:]
        # if k.startswith("module."):
        #     k = k[7:]
        if k.startswith("module.1."):
            k[0:9] = k[0:7]
        new_dict[k] = v
    return new_dict


def rand_normalize_directions(args, states, ignore='biasbn'):
    # assert(len(direction) == len(states))
    model = models.__dict__[args.arch]()
    init_dict = model.state_dict()
    new_dict = OrderedDict()
    for (k, w), (k2, d) in zip(states.items(), init_dict.items()):
        if w.dim() <= 1:
            if ignore == 'biasbn':
                d = torch.zeros_like(w)  # ignore directions for weights with 1 dimension
            else:
                d = w
        else:
        # if w.dim() > 1:
            d.mul_(w.norm()/(d.norm() + 1e-10))
        new_dict[k] = d
    return new_dict


def get_combined_weights(direction1, direction2, pretrained, weight1, weight2, weight_pretrained=1.0):
    new_dict = OrderedDict()
    for (k, d1),(_,d2), (_,w) in zip(direction1.items(), direction2.items(), pretrained.items()):
        new_dict[k] = (weight1 * d1 + weight2 * d2 + weight_pretrained * w)
        # new_dict[k] = (weight1 * d1 + weight2 * d2 + weight_pretrained * w)/\
        #               (abs(weight1)+abs(weight2)+abs(weight_pretrained))
    return new_dict


def main():
    args = parse_args()
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"



    assert torch.cuda.is_available(), "Please ensure codes are executed in cuda."
    torch.backends.cudnn.benchmark = True
    args.checkpoint = os.path.join("ablation_checkpoints",args.model+'-loss')


    logger = logging.getLogger("loss_landscape")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(args.checkpoint, "loss_landscape.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    list_1 = np.arange(-1, 1.1, 0.1)
    list_2 = np.arange(-1, 1.1, 0.1)

    checkpoint = load_pretrained(args)
    direction1 = rand_normalize_directions(args, checkpoint)
    direction2 = rand_normalize_directions(args, checkpoint)

    print('==> Building model..')
    net = models.__dict__[args.model]()
    criterion = cal_loss
    device = 'cuda'
    net = net.to(device)
    cudnn.benchmark = True

    print('==> Preparing data..')
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=args.workers,
                             batch_size=args.batch_size//2, shuffle=True, drop_last=False)


    for w1 in list_1:
        for w2 in list_2:
            print("\n\n===> w1 {w1:.2f} w2 {w2:.2f}".format(w1=w1, w2=w2))
            combined_weights = get_combined_weights(direction1, direction2, checkpoint, w1,w2)
            net.load_state_dict(combined_weights)
            test_out = validate(net, test_loader, criterion, device)
            logger.info("{w1:.2f},{w2:.2f},{loss},{accuracy}".
                        format(w1=w1, w2=w2,loss=test_out['loss'], accuracy=test_out['acc']))


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
