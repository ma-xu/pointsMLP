#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=ma.xu1@northeastern.edu
#SBATCH -N 1
#SBATCH -p fugpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32Gb
#SBATCH --output=%j.log

source /home/ma.xu1/.condarc
source activate point
cd /scratch/ma.xu1/pointsMLP/classification/
python voting.py --model model313C --msg 20220209053148-404 --NUM_PEPEAT 200
