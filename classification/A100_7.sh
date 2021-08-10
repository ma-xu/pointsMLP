#!/usr/bin/env bash

#python main.py --model model32K  --epoch 300 --msg epoch300_3
#python main.py --model model32L  --epoch 300 --msg epoch300_3
#python main.py --model model32M  --epoch 300 --msg epoch300_3
#python main.py --model model32N  --epoch 300 --msg epoch300_3


python main.py --model model31A --seed 42 --epoch 300 --msg epoch300_seed42
python main.py --model model31A --seed 1234 --epoch 300 --msg epoch300_seed1234
python main.py --model model31A --batch_size 16 --epoch 300 --msg epoch300_bs16_7
