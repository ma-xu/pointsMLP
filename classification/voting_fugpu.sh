#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=ma.xu1@northeastern.edu
#SBATCH -N 1
#SBATCH -p fugpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=%j.log

source /home/ma.xu1/.condarc
source activate point
cd /scratch/ma.xu1/pointsMLP/classification/
python voting.py --model model31C --msg 20210829112651 --epoch 200
