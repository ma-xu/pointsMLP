#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=ma.xu1@northeastern.edu
#SBATCH -N 1
#SBATCH -p fugpu
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:2
#SBATCH --output=%j.log

source activate point
cd /scratch/ma.xu1/pointsMLP/partseg_curvenet/
python main.py --model pformer34A --exp_name=Nov --workers 6
