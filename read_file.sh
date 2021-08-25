#!/usr/bin/env bash

# usage:
#  bash read_file.sh /scratch/ma.xu1/pointsMLP/classification/checkpoints/model31A-20210818204651
folder = $1

for file in $folder/*; do
    echo $file
done
