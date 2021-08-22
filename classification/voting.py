"""
nohup python eval_voting_enhance.py --model model21H --msg try2 > model21H_try2_evaluate_voting.out &
"""
import argparse
import os
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import models as models
from utils import Logger, mkdir_p, progress_bar, save_model, save_args, IOStream
from data import ModelNet40
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics
from helper import cal_loss
import numpy as np
import torch.nn.functional as F

model_names = sorted(name for name in models.__dict__
                     if callable(models.__dict__[name]))


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    # parser.add_argument('-d', '--data_path', default='data/modelnet40_normal_resampled/', type=str)
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='model31A', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_classes', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=350, type=int, help='number of epoch in training')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='decay rate')
    # parser.add_argument('--use_normals', action='store_true', default=False, help='use normals besides x,y,z')
    # parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    # parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampling')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    # Voting evaluation, referring: https://github.com/CVMI-Lab/PAConv/blob/main/obj_cls/eval_voting.py
    parser.add_argument('--NUM_PEPEAT', type=int, default=300)
    parser.add_argument('--NUM_VOTE', type=int, default=10)
    return parser.parse_args()


class PointcloudScale(object): # input random scaling
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda())

        return pc

def main():
    args = parse_args()
    print(f"args: {args}")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed(args.seed)
    else:
        device = 'cpu'
    print(f"==> Using device: {device}")
    if args.msg is None:
        message = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    else:
        message = "-"+args.msg
    args.checkpoint = 'checkpoints/' + args.model + message

    print('==> Preparing data..')
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.batch_size, shuffle=True, drop_last=False)
    vote_loader = DataLoader(ModelNet40(partition='test', num_points=2048), num_workers=8,
                             batch_size=args.batch_size, shuffle=True, drop_last=False)
    # Model
    print('==> Building model..')
    net = models.__dict__[args.model]()
    criterion = cal_loss
    net = net.to(device)
    checkpoint_path = os.path.join(args.checkpoint, 'best_checkpoint.pth')
    checkpoint = torch.load(checkpoint_path)
    # criterion = criterion.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net.load_state_dict(checkpoint['net'])
    test_out = validate(net, test_loader, criterion, device)
    print(f"Vanilla out: {test_out}")

    print(f"===> start voting evaluation...")
    voting(net, vote_loader, device, args)




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


def voting(net, testloader, device, args):
    name='/evaluate_voting'+str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))+'.log'
    io = IOStream(args.checkpoint + name)
    io.cprint(str(args))


    net.eval()
    best_acc = 0
    best_mean_acc = 0
    pointscale = PointcloudScale()  # set the range of scaling

    for i in range(args.NUM_PEPEAT):
        test_true = []
        test_pred = []

        for batch_idx, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            pred = 0
            for v in range(args.NUM_VOTE):
                idx = np.random.choice(2048, args.num_points, False)
                new_data = data[:, idx, :]
                # batch_size = data.size()[0]
                if v > 0:
                    new_data.data = pointscale(new_data.data)
                with torch.no_grad():
                    pred += F.softmax(net(new_data.permute(0, 2, 1)), dim=1)  # sum 10 preds
            pred /= args.NUM_VOTE  # avg the preds!
            label = label.view(-1)
            pred_choice = pred.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(pred_choice.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = 100. * metrics.accuracy_score(test_true, test_pred)
        test_mean_acc = 100. * metrics.balanced_accuracy_score(test_true, test_pred)
        if test_acc > best_acc:
            best_acc = test_acc
        if test_mean_acc > best_mean_acc:
            best_mean_acc = test_mean_acc
        outstr = 'Voting %d, test acc: %.3f, test mean acc: %.3f,  [current best(mean_acc: %.3f all_acc: %.3f)]' % \
                 (i, test_acc, test_mean_acc, best_acc, best_mean_acc)
        io.cprint(outstr)

    final_outstr = 'Final voting test acc: %.6f,' % (best_acc * 100)
    io.cprint(final_outstr)








if __name__ == '__main__':
    main()
