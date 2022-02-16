#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=ma.xu1@northeastern.edu
#SBATCH -N 1
#SBATCH -p multigpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32Gb
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=2
#SBATCH --output=out_%j.log
source /home/ma.xu1/.condarc
source activate point
cd /scratch/ma.xu1/pointsMLP/classification/
python voting.py --model model313C --msg 20220209053148 --NUM_PEPEAT 200
