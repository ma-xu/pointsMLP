#!/usr/bin/env bash


$dir = $1
cd $dir
filenames = $(ls *.log)

for file in ${filenames}; do
    echo ${file}
done
