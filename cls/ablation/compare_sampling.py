import argparse
import torch
import datetime
from pointnet2_ops import pointnet2_utils

def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--num_points', type=int, default=1024, help='Input Point Number')
    parser.add_argument('--sam_points', type=int, default=512, help='Sampling Point Number')
    parser.add_argument('--dim', type=int, default=3, help='xyz')
    parser.add_argument('--iterations', type=int, default=10000, help='xyz')
    parser.add_argument('--device', type=str, default="cuda")
    return parser.parse_args()


def measure_time(args):
    data = torch.rand(1,args.num_points, args.dim).to(args.device)
    warm_up=50
    for _ in range(warm_up):
        idx = torch.multinomial(torch.linspace(0, args.num_points - 1, steps=args.num_points).to(args.device),
                          num_samples=args.sam_points, replacement=False).long()
        idx = pointnet2_utils.furthest_point_sample(data, args.sam_points).long()
    time_cost_rand = datetime.datetime.now()
    for _ in range(args.iterations):
        idx = torch.multinomial(torch.linspace(0, args.num_points - 1, steps=args.num_points).to(args.device),
                                num_samples=args.sam_points, replacement=False).long()
    time_cost_rand = (datetime.datetime.now() - time_cost_rand).total_seconds()

    time_cost_fps = datetime.datetime.now()
    for _ in range(args.iterations):
        idx = pointnet2_utils.furthest_point_sample(data, args.sam_points).long()
    time_cost_fps = (datetime.datetime.now() - time_cost_fps).total_seconds()
    print(f'\t[points:{args.num_points}, sampling:{args.sam_points}, iterations:{args.iterations}, device:{args.device}]'
          f'  FPS time: {time_cost_fps} | RAND time: {time_cost_rand}')


if __name__ == '__main__':
    args = parse_args()
    measure_time(args)
