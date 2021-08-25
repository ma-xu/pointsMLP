#!/usr/bin/env bash


$dir = $1
cd $dir
$filenames = $(ls *.log)
echo $filenames

for file in ${filenames}; do
    echo ${file}
done
