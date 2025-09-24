# seed=$1

# python3 train_minigrid_ppo.py --env darkroom_heldout --envs 40 --H 100 --dim 10 --fix_seed --seed $seed


#!/bin/bash

# for seed in {11..15}; do
# for seed in 18 25 50 58 73; do
for seed in 3 5 6 7 8 10 13 14 16 17; do
    # 使用nohup后台运行，日志保存到单独文件
    nohup python3 train_minigrid_ppo.py \
        --env darkroom_heldout \
        --envs 10 \
        --H 100 \
        --dim 10 \
        --fix_seed \
        --seed $seed > "logs/MiniGrid/nohup_seed${seed}.log" 2>&1 &
done