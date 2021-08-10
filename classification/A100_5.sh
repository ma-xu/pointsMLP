#!/usr/bin/env bash

#python main.py --model model32K  --epoch 300 --msg epoch300_1
#python main.py --model model32L  --epoch 300 --msg epoch300_1
#python main.py --model model32M  --epoch 300 --msg epoch300_1
#python main.py --model model32N  --epoch 300 --msg epoch300_1


python main.py --model model31A --seed 100 --epoch 300 --msg epoch300_seed100
python main.py --model model31A --seed 666 --epoch 300 --msg epoch300_seed666
python main.py --model model31A --batch_size 16 --epoch 300 --msg epoch300_bs16_5
