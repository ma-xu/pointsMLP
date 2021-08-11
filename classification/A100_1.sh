#!/usr/bin/env bash

#python main.py --model model32D  --epoch 300 --msg epoch300_1
#python main.py --model model32E  --epoch 300 --msg epoch300_1
#python main.py --model model32F  --epoch 300 --msg epoch300_1
#python main.py --model model32G  --epoch 300 --msg epoch300_1


#python main.py --model model31A --seed 0 --epoch 300 --msg epoch300_seed0
python main.py --model model31A --seed 1 --epoch 300 --msg epoch300_seed1
python main.py --model model31A --batch_size 16 --epoch 300 --msg epoch300_bs16_1
python main.py --model model33A --epoch 300 --msg epoch300_1
python main.py --model model33H --epoch 300 --msg epoch300_2
