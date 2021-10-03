#!/bin/bash
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=ma.xu1@northeastern.edu
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:t4:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32Gb
#SBATCH --time=8:00:00
#SBATCH --output=%j.log
source activate point
cd /scratch/ma.xu1/pointsMLP/classification/
python voting.py --model modelelite3X10 --msg 20211002174353 --epoch 200
