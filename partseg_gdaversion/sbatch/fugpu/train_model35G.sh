#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=ma.xu1@northeastern.edu
#SBATCH -N 1
#SBATCH -p fugpu
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:4
#SBATCH --output=%j.log

source activate point
cd /scratch/ma.xu1/pointsMLP/partseg_gdaversion/
python main.py --model model35G --exp_name cos_bs32 --scheduler cos --epochs 250 &
python main.py --model model35A --exp_name cos_bs32 --scheduler cos --epochs 250 &
