#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=ma.xu1@northeastern.edu
#SBATCH -N 1
#SBATCH -p fugpu
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --output=%j.log

source activate point
cd /scratch/ma.xu1/pointsMLP/partseg_v2/
python main.py --model model40A --scheduler cos --batch_size 32
