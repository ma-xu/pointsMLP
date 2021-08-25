#!/usr/bin/env bash

echo $1
cd $1
filenames=$(ls *.log)

for file in ${filenames}; do
    echo ${file}
done
