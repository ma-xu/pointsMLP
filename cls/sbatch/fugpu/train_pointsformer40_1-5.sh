#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=ma.xu1@northeastern.edu
#SBATCH -N 1
#SBATCH -p fugpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --output=%j.log

source activate point
cd /scratch/ma.xu1/pointsMLP/cls/
python main.py --model pointsformer1 --scheduler cos --batch_size 32 --workers 4
python main.py --model pointsformer2 --scheduler cos --batch_size 32 --workers 4
python main.py --model pointsformer3 --scheduler cos --batch_size 32 --workers 4
python main.py --model pointsformer4 --scheduler cos --batch_size 32 --workers 4
python main.py --model pointsformer5 --scheduler cos --batch_size 32 --workers 4
