# seed=$1

# python3 eval_minigrid_ppo.py --env darkroom_heldout --envs 40 --H 100 --dim 10 --fix_seed --seed $seed

python3 eval_minigrid_ppo.py --env darkroom_heldout --envs 40 --H 100 --dim 10 --multi_seed