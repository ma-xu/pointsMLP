#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=ma.xu1@northeastern.edu
#SBATCH -N 1
#SBATCH -p fugpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=%j.log

source activate point
cd /scratch/ma.xu1/pointsMLP/classification/
python main.py --model model31CNoAffine --epoch 300 --workers 4  --weight_decay 2e-4
