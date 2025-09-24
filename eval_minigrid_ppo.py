from envs import darkroom_env, bandit_env
from envs.darkroom_env import DarkroomEnvVec
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

import wandb
from wandb.integration.sb3 import WandbCallback
import minigrid

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
from minigrid.wrappers import ImgObsWrapper
from loguru import logger
import pprint
from env_list import ENVS, HARD_ENVS
# from minigrid.vecenv import MiniGridVecEnv
# import stable_baselines3

# ENV = "MiniGrid-LavaCrossingS9N3-v0"
# ENV = "MiniGrid-RedBlueDoors-8x8-v0"
# ENV = "MiniGrid-UnlockPickup-v0"
# ENV = "MiniGrid-BlockedUnlockPickup-v0"
ENV = "MiniGrid-LavaCrossingS11N5-v0"
# ENV = "MiniGrid-FourRooms-v0"


def cal_best_traj(env):
    
    return

def make_eval_env(seed=-1):
    env = gym.make(ENV, render_mode="rgb_array")
    env = ImgObsWrapper(env)
    if seed != -1:
        logger.info(f"set seed to {seed}")
        env.reset(seed=seed)
    else:
        env.reset()

    return env

def eval_multi_seed(model, env, seed_list):
    seed_res = []
    for seed in seed_list:
        terminated = False
        truncated = False
        total_reward = 0
        num_trajs = 5
        grid = env.unwrapped.pprint_grid()
        # pprint.pprint(grid)
        for _ in range(num_trajs):
            obs, _ = env.reset(seed=seed)
            # logger.info(f"obs shape: {obs.shape}")
            eps_reward = 0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=False)
                new_obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                eps_reward += reward
                obs = new_obs

                # if (terminated or truncated):
                #     if eps_reward == 0:
                #         logger.info(f"Encounter 0 reward at seed {seed}!")
                        # pprint.pprint(grid)
        logger.info(f"Episode reward for seed {seed}: {total_reward / num_trajs}")
        seed_res.append(total_reward / num_trajs)
        if total_reward / num_trajs < 0.9:
            logger.info(f"Low reward for seed {seed}")

    # get the lowest 100 seed
    seed_res = np.array(seed_res)
    # logger.info(f"Average episode reward for all seeds: {np.mean(seed_res)}")
    logger.info(f"Lowest 100 seeds: {np.argsort(seed_res)[:100]}")
    logger.info(f"Lowest 100 seeds rewards: {seed_res[np.argsort(seed_res)[:100]]}")



def sample_trajectories(model, env, num_trajs=5):
    trajectories = []
    total_reward = 0
    # grid = env.unwrapped.pprint_grid()
    # pprint.pprint(grid)
    for _ in range(num_trajs):
        obs, _ = env.reset()
        episode_obs = []
        episode_actions = []
        episode_rewards = []
        terminated = False
        truncated = False
        step = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=False)
            new_obs, reward, terminated, truncated, info = env.step(action)
            
            # 记录轨迹数据
            episode_obs.append(obs)
            episode_actions.append(action)
            episode_rewards.append(reward)
            total_reward += reward
            
            obs = new_obs
            step += 1

        # 将列表转换为numpy数组
        trajectory = {
            "observations": np.stack(episode_obs),
            "actions": np.array(episode_actions),
            "rewards": np.array(episode_rewards)
        }
        trajectories.append(trajectory)
    
    logger.info(f"Average episode reward: {total_reward / num_trajs}")
    return trajectories


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
    parser.add_argument("--fix_seed", action="store_true",
                        default=False, help="Fix seed or not")
    parser.add_argument("--multi_seed", action="store_true",
                        default=False, help="Fix seed or not")
    
    parser.add_argument("--seed", type=int, default=128)
    args = vars(parser.parse_args())
    print("Args: ", args)

    assert not args['fix_seed'] or not args['multi_seed'], "Cannot fix seed while multi seed"

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

    n_train_envs = int(.8 * n_envs)
    n_test_envs = n_envs - n_train_envs

    config = {
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
    }

    config.update({'dim': dim, 'rollin_type': 'uniform'})

    # test_env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
    # print(test_env)
    seed = args['seed']
    if not args['fix_seed']:
        logger.info(f"no fix seed setting")
        env = make_eval_env()
    else:
        logger.info(f"set seed to {seed}")
        env = make_eval_env(seed=seed)

    # model = PPO("CnnPolicy", env, verbose=2)
    # if not args['fix_seed']:
    #     model = PPO.load(f"{ENV}_noseed", env=env)
    # else:
    #     model = PPO.load(f"{ENV}_{seed}", env=env)
    model = PPO.load(f"models/MiniGrid/{ENV}/noseed_cnn.zip", env=env)

    if not args["multi_seed"]:
        sample_trajectories(model, env)
    else: # multi-seed
        seed_list = [i+1 for i in range(0, 1000)]
        eval_multi_seed(model, env, seed_list)





