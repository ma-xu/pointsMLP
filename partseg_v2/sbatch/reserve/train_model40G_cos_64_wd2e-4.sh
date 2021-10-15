#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=ma.xu1@northeastern.edu
#SBATCH -N 1
#SBATCH -p reservation
#SBATCH --reservation=xuma_mlp_test
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100-sxm2:2
#SBATCH --cpus-per-task=12
#SBATCH --output=%j.log

source activate point
cd /scratch/ma.xu1/pointsMLP/partseg_v2/
python main.py --model model40G --exp_name cos_bs64_wd2e-4 --scheduler cos --batch_size 64 --weight_decay 2e-4
