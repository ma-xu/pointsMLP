#!/usr/bin/env bash

#python main.py --model model32K  --epoch 300 --msg epoch300_2
#python main.py --model model32L  --epoch 300 --msg epoch300_2
#python main.py --model model32M  --epoch 300 --msg epoch300_2
#python main.py --model model32N  --epoch 300 --msg epoch300_2


#python main.py --model model31A --seed 10086 --epoch 300 --msg epoch300_seed10086
python main.py --model model31A --seed 99 --epoch 300 --msg epoch300_seed99
python main.py --model model31A --batch_size 16 --epoch 300 --msg epoch300_bs16_6
python main.py --model model33F --epoch 300 --msg epoch300_1
python main.py --model model33C --epoch 300 --msg epoch300_2
