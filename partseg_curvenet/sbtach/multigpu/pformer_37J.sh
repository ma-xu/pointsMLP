#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=ma.xu1@northeastern.edu
#SBATCH -N 1
#SBATCH -p multigpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128Gb
#SBATCH --gres=gpu:v100-sxm2:2
#SBATCH --time=1-00:00:00
#SBATCH --output=%j.log

source activate point
cd /scratch/ma.xu1/pointsMLP/partseg_curvenet/
python main.py --model pformer37J --exp_name=Oct30multi --workers 8
