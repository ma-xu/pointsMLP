#!/bin/bash
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=ma.xu1@northeastern.edu
#SBATCH -N 1
#SBATCH -p multigpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32Gb
#SBATCH --time=1-00:00:00
#SBATCH --output=%j.log
source activate point
cd /scratch/ma.xu1/pointsMLP/classification/
python voting.py --model modelelite3X10 --msg 20211002174353 --NUM_PEPEAT 200
python voting.py --model modelelite3X10 --msg 20211002174353 --NUM_PEPEAT 200
python voting.py --model modelelite3X10 --msg 20211002174353 --NUM_PEPEAT 200
