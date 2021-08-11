#!/usr/bin/env bash

#python main.py --model model32D  --epoch 300 --msg epoch300_4
#python main.py --model model32E  --epoch 300 --msg epoch300_4
#python main.py --model model32F  --epoch 300 --msg epoch300_4
#python main.py --model model32G  --epoch 300 --msg epoch300_4


#python main.py --model model31A --seed 6 --epoch 300 --msg epoch300_seed6
python main.py --model model31A --seed 7 --epoch 300 --msg epoch300_seed7
python main.py --model model31A --batch_size 16 --epoch 300 --msg epoch300_bs16_4
python main.py --model model33D --epoch 300 --msg epoch300_1
python main.py --model model33E --epoch 300 --msg epoch300_2
