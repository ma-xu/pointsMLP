#!/usr/bin/env bash

#python main.py --model model32K  --epoch 300 --msg epoch300_4
#python main.py --model model32L  --epoch 300 --msg epoch300_4
#python main.py --model model32M  --epoch 300 --msg epoch300_4
#python main.py --model model32N  --epoch 300 --msg epoch300_4


#python main.py --model model31A --seed 111 --epoch 300 --msg epoch300_seed111
python main.py --model model31A --seed 321 --epoch 300 --msg epoch300_seed321
python main.py --model model31A --batch_size 16 --epoch 300 --msg epoch300_bs16_8
python main.py --model model33H --epoch 300 --msg epoch300_1
python main.py --model model33A --epoch 300 --msg epoch300_2
