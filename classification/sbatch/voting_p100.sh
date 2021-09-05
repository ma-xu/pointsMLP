#!/bin/bash
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=ma.xu1@northeastern.edu
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:p100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16Gb
#SBATCH --time=08:00:00
#SBATCH --output=../nohup/%j.log
source activate point
cd /scratch/ma.xu1/pointsMLP/classification/

python voting.py --model model31C --msg 20210829112651 --NUM_PEPEAT 100
