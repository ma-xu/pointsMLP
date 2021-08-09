#!/usr/bin/env bash

python main.py --model model32K  --epoch 300 --msg epoch300_1
python main.py --model model32L  --epoch 300 --msg epoch300_1
python main.py --model model32M  --epoch 300 --msg epoch300_1
python main.py --model model32N  --epoch 300 --msg epoch300_1
