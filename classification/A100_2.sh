#!/usr/bin/env bash

#python main.py --model model32D  --epoch 300 --msg epoch300_2
#python main.py --model model32E  --epoch 300 --msg epoch300_2
#python main.py --model model32F  --epoch 300 --msg epoch300_2
#python main.py --model model32G  --epoch 300 --msg epoch300_2

python main.py --model model31A --seed 2 --epoch 300 --msg epoch300_seed2
python main.py --model model31A --seed 3 --epoch 300 --msg epoch300_seed3
python main.py --model model31A --batch_size 16 --epoch 300 --msg epoch300_bs16_2
