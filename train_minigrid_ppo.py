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
from env_list import ENVS, HARD_ENVS
from nets.custom_net import CustomCNN
from stable_baselines3.common.callbacks import CheckpointCallback
# from minigrid.vecenv import MiniGridVecEnv
# import stable_baselines3

# ENV = "MiniGrid-FourRooms-v0"
ENV = "MiniGrid-LavaCrossingS9N3-v0"
# ENV = "MiniGrid-BlockedUnlockPickup-v0"
# ENV = "MiniGrid-UnlockPickup-v0"
# ENV = "MiniGrid-Unlock-v0"
# ENV = "MiniGrid-SimpleCrossingS9N3-v0"
# ENV = "MiniGrid-SimpleCrossingS11N5-v0"
# ENV = "MiniGrid-LavaCrossingS11N5-v0"s

def make_env(seed=-1):  
    def _init():  
        # env = darkroom_env.DarkroomEnv(dim, goal, horizon)
        env = gym.make(ENV, render_mode="rgb_array")
        env = ImgObsWrapper(env)
        if seed != -1:
            logger.info(f"set seed to {seed}")
            env.reset(seed=seed)
        else:
            env.reset()
        return Monitor(env)
        # return env
    return _init  

def make_minigrid_env(num_envs, seed=-1):
    envs = [make_env(seed=seed) for _ in range(num_envs)]
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
    parser.add_argument("--fix_seed", action="store_true",
                        default=False, help="Fix seed or not")
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--continue", action="store_true",
                        default=False, help="Continue train or not")
    parser.add_argument("--finetune", action="store_true",
                        default=False, help="Finetune from noseed or not")
    parser.add_argument("--cnn", action="store_true",
                        default=False, help="Use CnnPolicy or not")
    args = vars(parser.parse_args())
    print("Args: ", args)

    assert not args["continue"] or not args["finetune"], "cannot both continue and finetune"
    assert not args["finetune"] or args["fix_seed"], "when finetune, you must fix seed"

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

    n_train_envs = int(n_envs)
    n_test_envs = int(.2 * n_envs)

    config = {
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
    }

    seed = args['seed']
    if not args['fix_seed']:
        wandb_name = f"{ENV}-ppo-noseed"
        wandb_group = f"{ENV}-ppo-noseed"
    elif args['finetune']:
        wandb_name = f"{ENV}-ppo-finetune-seed{args['seed']}"
        wandb_group = f"{ENV}-ppo-fintune"
    else:
        wandb_name = f"{ENV}-ppo-seed{args['seed']}"
        wandb_group = f"{ENV}-ppo-fixseed"
    run = wandb.init(
        project="minigrid-ppo",
        name=wandb_name,
        group=wandb_group,
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
    run_id = run.id

    config.update({'dim': dim, 'rollin_type': 'uniform'})

    # test_env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
    # print(test_env)
    if not args['fix_seed']:
        logger.info(f"no fix seed setting")
        env = make_minigrid_env(n_train_envs)
        eval_env = make_minigrid_env(n_test_envs)
    else:
        logger.info(f"set seed to {seed}")
        env = make_minigrid_env(n_train_envs, seed=seed)
        eval_env = make_minigrid_env(n_test_envs, seed=seed)

    if args["cnn"]:
        if args['continue']:
            save_path = f"models/MiniGrid/{ENV}/noseed_continue_cnn"
        elif not args['fix_seed']:
            save_path = f"models/MiniGrid/{ENV}/noseed_cnn"
        elif args['finetune']:
            save_path = f"models/MiniGrid/{ENV}/{seed}-funetune_cnn"
        else:
            save_path = f"models/MiniGrid/{ENV}/{seed}_cnn"
    else:
        if args['continue']:
            save_path = f"models/MiniGrid/{ENV}/noseed_continue"
        elif not args['fix_seed']:
            save_path = f"models/MiniGrid/{ENV}/noseed"
        elif args['finetune']:
            save_path = f"models/MiniGrid/{ENV}/{seed}-funetune"
        else:
            save_path = f"models/MiniGrid/{ENV}/{seed}"

    os.makedirs(f"./{save_path}/run_{run_id}/logs", exist_ok=True)
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                 log_path=f"./{save_path}/run_{run_id}/logs/", eval_freq=2000,
                                 deterministic=True, render=False)
    save_freq = 1600
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=f"./{save_path}/run_{run_id}/",
        name_prefix="ppo",
        verbose=2,
    )
    reward_callback = RewardWandbCallback(
        verbose=2,
    )

    if args["continue"]:
        logger.info("\n\n\n ======= CONTINUE TRAINING FROM NOSEED ======= \n\n\n")
        model = PPO.load(f"models/MiniGrid/{ENV}/noseed.zip", env=env)
    elif args["finetune"]:
        logger.info("\n\n\n ======= FINETUNING FROM NOSEED ======= \n\n\n")
        if args["cnn"]:
            model = PPO.load(f"models/MiniGrid/{ENV}/noseed_cnn.zip", env=env, learning_rate=2e-5, n_epochs=10)
        else:
            model = PPO.load(f"models/MiniGrid/{ENV}/noseed.zip", env=env, learning_rate=2e-5, n_epochs=10)
    else:
        if args["cnn"]:
            logger.info("USE CNNPOLICY")
            policy_kwargs = dict(
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=128),
            )
            if ENV == "MiniGrid-LavaCrossingS11N5-v0":
                logger.info("very hard, try to finetune from S9N3")
                model = PPO.load(f"models/MiniGrid/MiniGrid-LavaCrossingS9N1-v0/noseed_cnn.zip", env=env, learning_rate=5e-5)
            elif ENV == "MiniGrid-RedBlueDoors-8x8-v0":
                logger.info("very hard, try to finetune from 6x6")
                model = PPO.load(f"models/MiniGrid/MiniGrid-RedBlueDoors-6x6-v0/noseed_cnn.zip", env=env, learning_rate=5e-5)
            elif ENV == "MiniGrid-BlockedUnlockPickup-v0":
                logger.info("very hard, try to finetune from UnlockPickup")
                model = PPO.load(f"models/MiniGrid/MiniGrid-UnlockPickup-v0/noseed_cnn.zip", env=env, learning_rate=5e-5)
            elif ENV in HARD_ENVS:
                model = PPO("CnnPolicy", env, verbose=2, device='cuda', n_steps=1600, learning_rate=2e-4, n_epochs=20, policy_kwargs=policy_kwargs)
            else:
                model = PPO("CnnPolicy", env, verbose=2, device='cuda', n_steps=1600, learning_rate=1e-4, policy_kwargs=policy_kwargs)
        else:
            logger.info("USE MLPPOLICY")
            if ENV in HARD_ENVS:
                model = PPO("MlpPolicy", env, verbose=2, device='cuda', n_steps=1600, learning_rate=2e-4, n_epochs=20)
            else:
                model = PPO("MlpPolicy", env, verbose=2, device='cuda', n_steps=1600, learning_rate=1e-4)

    if not args['fix_seed']:
        if ENV in HARD_ENVS:
            model.learn(total_timesteps=2e7, callback=[checkpoint_callback, reward_callback])
        else:
            model.learn(total_timesteps=4000000, callback=[checkpoint_callback, reward_callback])
    else:
        if ENV in HARD_ENVS:
            model.learn(total_timesteps=400000, callback=[checkpoint_callback, reward_callback])
        else:
            model.learn(total_timesteps=400000, callback=[checkpoint_callback, reward_callback])
    
    model.save(f"{save_path}.zip")
    
    model.save(f"{save_path}/run_{run_id}/final")





