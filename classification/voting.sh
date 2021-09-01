#!/bin/bash
#SBATCH --job-name=vote_model31C_20210829112651
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=ma.xu1@northeastern.edu
#SBATCH -N 1
#SBATCH -p multigpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32Gb
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=2
#SBATCH --output=nohup/out_%j.log
conda activate point
cd /scratch/ma.xu1/pointsMLP/classification/
python voting.py --model model31C --msg 20210829112651 --epoch 200
