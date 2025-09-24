from envs import darkroom_env, bandit_env
from envs.darkroom_env import DarkroomEnvVec
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

import wandb
from wandb.integration.sb3 import WandbCallback

import argparse
import os
import pickle
import random

import gymnasium as gym # gym
import numpy as np
from skimage.transform import resize
from IPython import embed

import common_args
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.callbacks import CheckpointCallback
# import stable_baselines3


def make_env(dim, goal, horizon):  
    def _init():  
        env = darkroom_env.DarkroomEnv(dim, goal, horizon)
        return Monitor(env)
        # return env
    return _init  

def make_darkroom_env(goal, dim, horizon, num_envs):
    envs = [make_env(dim, goal, horizon) for _ in range(num_envs)]
    env = DummyVecEnv(envs)
    return env

class RewardWandbCallback(WandbCallback):  
    def _on_step(self) -> bool:  
        # 获取 Monitor 收集的奖励等信息  
        infos = self.locals.get("infos", [])  
        for info in infos:  
            if "episode" in info.keys():  
                wandb.log({"eps_reward": info["episode"]["r"]})  # 记录到 wandb  
        return super()._on_step()  


if __name__ == '__main__':

    np.random.seed(0)
    random.seed(0)

    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    parser.add_argument('--goal_x', type=int, default=5, help='goal x coordinate')
    parser.add_argument('--goal_y', type=int, default=5, help='goal y coordinate')
    args = vars(parser.parse_args())
    print("Args: ", args)

    env = args['env']
    n_envs = args['envs']
    n_eval_envs = args['envs_eval']
    n_hists = args['hists']
    n_samples = args['samples']
    horizon = args['H']
    dim = args['dim']
    var = args['var']
    cov = args['cov']
    env_id_start = args['env_id_start']
    env_id_end = args['env_id_end']
    lin_d = args['lin_d']
    goal_x = args['goal_x']
    goal_y = args['goal_y']

    n_train_envs = n_envs
    n_test_envs = int(0.2 * n_envs)

    config = {
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
    }
    wandb_name = f"darkroom-ppo-dim{dim}-{goal_x}-{goal_y}"
    wandb_group = f"darkroom-ppo-dim{dim}"

    run = wandb.init(
        project="darkroom-ppo",
        name=wandb_name,
        group=wandb_group,
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
    run_id = run.id

    save_path = f"models/Darkroom/{goal_x}-{goal_y}"

    save_freq = 500
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=f"./{save_path}/run_{run_id}/",
        name_prefix="ppo",
        verbose=2,
    )
    reward_callback = RewardWandbCallback(
        verbose=2,
    )

    config.update({'dim': dim, 'rollin_type': 'uniform'})
    goal = (goal_x, goal_y)

    env = make_darkroom_env(goal, dim, horizon, n_train_envs)
    eval_env = make_darkroom_env(goal, dim, horizon, n_test_envs)

    model = PPO("MlpPolicy", env, verbose=2, device='cuda', n_steps=500, learning_rate=1e-4)
    model.learn(total_timesteps=1000000, callback=[checkpoint_callback, reward_callback])

    model.save(f"./{save_path}/run_{run_id}/final")





