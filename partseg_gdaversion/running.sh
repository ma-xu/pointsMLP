#!/usr/bin/env bash

python main.py --model model31G --exp_name exp1 --batch_size 32 --scheduler cos
python main.py --model model31G --exp_name exp2 --batch_size 32 --scheduler cos
python main.py --model model31G --exp_name exp3 --batch_size 32 --scheduler cos
python main.py --model model32G --exp_name exp1 --batch_size 32 --scheduler cos
python main.py --model model32G --exp_name exp2 --batch_size 32 --scheduler cos
python main.py --model model32G --exp_name exp3 --batch_size 32 --scheduler cos
python main.py --model model32G --exp_name exp4 --batch_size 32 --scheduler cos
