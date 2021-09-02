#!/bin/bash
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=ma.xu1@northeastern.edu
#SBATCH -N 1
#SBATCH -p multigpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32Gb
#SBATCH --time=1-00:00:00
#SBATCH --output=../nohup/out_%j.log

source activate point
cd /scratch/ma.xu1/pointsMLP/classification/
python main_pointsformer.py --model model31C --epoch 300 --workers 4 --optimizer adam --weight_decay 0.0003
