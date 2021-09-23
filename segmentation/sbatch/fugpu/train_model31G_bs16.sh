#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=ma.xu1@northeastern.edu
#SBATCH -N 1
#SBATCH -p fugpu
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=%j.log

source activate point
cd /scratch/ma.xu1/pointsMLP/segmentation/
python main.py --model model31G --batch_size 16 --learning_rate 0.001 --optimizer Adam --scheduler cos --workers 12 &
