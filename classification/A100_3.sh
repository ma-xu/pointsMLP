#!/usr/bin/env bash
#
#python main.py --model model32D  --epoch 300 --msg epoch300_3
#python main.py --model model32E  --epoch 300 --msg epoch300_3
#python main.py --model model32F  --epoch 300 --msg epoch300_3
#python main.py --model model32G  --epoch 300 --msg epoch300_3

#python main.py --model model31A --seed 4 --epoch 300 --msg epoch300_seed4
python main.py --model model31A --seed 5 --epoch 300 --msg epoch300_seed5
python main.py --model model31A --batch_size 16 --epoch 300 --msg epoch300_bs16_3
python main.py --model model33C --epoch 300 --msg epoch300_1
python main.py --model model33F --epoch 300 --msg epoch300_2
