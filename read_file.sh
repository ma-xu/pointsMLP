#!/usr/bin/env bash

# usage:
#  bash read_file.sh /scratch/ma.xu1/pointsMLP/classification/checkpoints/model31A-20210818204651
dir = /scratch/ma.xu1/pointsMLP/classification/checkpoints/model31A-20210818204651

for file in $dir/*; do
    echo $file
done
