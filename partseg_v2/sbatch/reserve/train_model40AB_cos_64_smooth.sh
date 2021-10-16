#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=ma.xu1@northeastern.edu
#SBATCH -N 1
#SBATCH -p reservation
#SBATCH --reservation=xuma_mlp_test
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100-sxm2:4
#SBATCH --cpus-per-task=24
#SBATCH --output=%j.log

source activate point
cd /scratch/ma.xu1/pointsMLP/partseg_v2/
CUDA_VISIBLE_DEVICES=0,1 python main.py --model model40A --scheduler cos --batch_size 64 --smooth True &
CUDA_VISIBLE_DEVICES=2,3 python main.py --model model40B --scheduler cos --batch_size 64 --smooth True
