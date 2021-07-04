import argparse
import torch
import numpy as np
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
    warm_up=10
    for _ in range(warm_up):
        idx = torch.Tensor(np.random.choice(args.num_points,size=args.num_points, replace=False)).to(args.device).long()
        idx = pointnet2_utils.furthest_point_sample(data, args.sam_points).long()
    print("Finishing warm up devices")
    time_cost_rand = datetime.datetime.now()
    for _ in range(args.iterations):
        idx = torch.Tensor(np.random.choice(args.num_points,size=args.num_points, replace=False)).to(args.device).long()
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


# salloc -N 1 -p multigpu --gres=gpu:v100-sxm2:1 --cpus-per-task=1 --mem=128Gb  --time=1-00:00:00
# results
# (point) [ma.xu1@d1009 ablation]$ python compare_sampling.py --num_points 1024 --sam_points 512 --iterations 500 --device cuda
# Finishing warm up devices
# 	[points:1024, sampling:512, iterations:500, device:cuda]  FPS time: 0.17862 | RAND time: 0.035436
# (point) [ma.xu1@d1009 ablation]$
# (point) [ma.xu1@d1009 ablation]$ python compare_sampling.py --num_points 2048 --sam_points 1024 --iterations 500 --device cuda
# Finishing warm up devices
# 	[points:2048, sampling:1024, iterations:500, device:cuda]  FPS time: 0.360224 | RAND time: 0.043737
# (point) [ma.xu1@d1009 ablation]$ python compare_sampling.py --num_points 4096 --sam_points 2048 --iterations 500 --device cuda
# Finishing warm up devices
# 	[points:4096, sampling:2048, iterations:500, device:cuda]  FPS time: 0.915579 | RAND time: 0.061251
# (point) [ma.xu1@d1009 ablation]$ python compare_sampling.py --num_points 8192 --sam_points 4096 --iterations 500 --device cuda
# Finishing warm up devices
# 	[points:8192, sampling:4096, iterations:500, device:cuda]  FPS time: 4.47864 | RAND time: 0.106079
# (point) [ma.xu1@d1009 ablation]$ python compare_sampling.py --num_points 16384 --sam_points 8192 --iterations 500 --device cuda
# Finishing warm up devices
# 	[points:16384, sampling:8192, iterations:500, device:cuda]  FPS time: 16.085766 | RAND time: 0.211724
# (point) [ma.xu1@d1009 ablation]$ python compare_sampling.py --num_points 32768 --sam_points 16384 --iterations 500 --device cuda
# Finishing warm up devices
# 	[points:32768, sampling:16384, iterations:500, device:cuda]  FPS time: 59.364385 | RAND time: 0.510331
# (point) [ma.xu1@d1009 ablation]$ python compare_sampling.py --num_points 65536 --sam_points 32768 --iterations 500 --device cuda
# Finishing warm up devices
# 	[points:65536, sampling:32768, iterations:500, device:cuda]  FPS time: 227.105271 | RAND time: 1.44456
# (point) [ma.xu1@d1009 ablation]$
#
