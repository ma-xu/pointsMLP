#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=ma.xu1@northeastern.edu
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32Gb
#SBATCH --time=8:00:00
#SBATCH --output=%j.log

source activate point
cd /scratch/ma.xu1/pointsMLP/partseg_gdaversion/
python vote_ptseg.py --model model40G --exp_name cos_bs32 --scheduler cos --batch_size 32
