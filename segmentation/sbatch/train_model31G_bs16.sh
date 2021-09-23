#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ma.xu1@northeastern.edu
#SBATCH -N 1
#SBATCH -p multigpu
#SBATCH --gres=gpu:v100-sxm2:4
#SBATCH --cpus-per-task=28
#SBATCH --mem=128Gb
#SBATCH --time=1-00:00:00
#SBATCH --output=%j.log

source activate point
cd /scratch/ma.xu1/pointsMLP/segmentation/
CUDA_VISIBLE_DEVICES=0,1 python main.py --model model31G --batch_size 16 --learning_rate 0.001 --optimizer Adam --scheduler cos --exp_name bs32lr0.001 --workers 28 &
CUDA_VISIBLE_DEVICES=2,3 python main.py --model model31G --batch_size 16 --learning_rate 0.003 --optimizer Adam --scheduler cos --exp_name bs32lr0.003 --workers 28

