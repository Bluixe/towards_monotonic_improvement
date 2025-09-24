import pickle

import numpy as np
import torch
from loguru import logger
from utils import convert_to_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataset(torch.utils.data.Dataset):
    """Dataset class."""

    def __init__(self, path, config):
        self.shuffle = config['shuffle']
        self.horizon = config['horizon']
        self.store_gpu = config['store_gpu']
        self.config = config

        # if path is not a list
        if not isinstance(path, list):
            path = [path]

        self.trajs = []
        for p in path:
            with open(p, 'rb') as f:
                self.trajs += pickle.load(f)
            
        context_states = []
        context_actions = []
        context_next_states = []
        context_rewards = []
        query_states = []
        optimal_actions = []

        for traj in self.trajs:
            context_states.append(traj['context_states'])
            context_actions.append(traj['context_actions'])
            context_next_states.append(traj['context_next_states'])
            context_rewards.append(traj['context_rewards'])

            query_states.append(traj['query_state'])
            optimal_actions.append(traj['optimal_action'])

        context_states = np.array(context_states)
        context_actions = np.array(context_actions)
        context_next_states = np.array(context_next_states)
        context_rewards = np.array(context_rewards)
        if len(context_rewards.shape) < 3:
            context_rewards = context_rewards[:, :, None]
        query_states = np.array(query_states)
        optimal_actions = np.array(optimal_actions)

        self.dataset = {
            'query_states': convert_to_tensor(query_states, store_gpu=self.store_gpu),
            'optimal_actions': convert_to_tensor(optimal_actions, store_gpu=self.store_gpu),
            'context_states': convert_to_tensor(context_states, store_gpu=self.store_gpu),
            'context_actions': convert_to_tensor(context_actions, store_gpu=self.store_gpu),
            'context_next_states': convert_to_tensor(context_next_states, store_gpu=self.store_gpu),
            'context_rewards': convert_to_tensor(context_rewards, store_gpu=self.store_gpu),
        }

        self.zeros = np.zeros(
            config['state_dim'] ** 2 + config['action_dim'] + 1
        )
        self.zeros = convert_to_tensor(self.zeros, store_gpu=self.store_gpu)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset['query_states'])

    def __getitem__(self, index):
        'Generates one sample of data'
        res = {
            'context_states': self.dataset['context_states'][index],
            'context_actions': self.dataset['context_actions'][index],
            'context_next_states': self.dataset['context_next_states'][index],
            'context_rewards': self.dataset['context_rewards'][index],
            'query_states': self.dataset['query_states'][index],
            'optimal_actions': self.dataset['optimal_actions'][index],
            'zeros': self.zeros,
        }

        if self.shuffle:
            perm = torch.randperm(self.horizon)
            res['context_states'] = res['context_states'][perm]
            res['context_actions'] = res['context_actions'][perm]
            res['context_next_states'] = res['context_next_states'][perm]
            res['context_rewards'] = res['context_rewards'][perm]

        return res
    

class ImageDataset(Dataset):
    """"Old image dataset class for image-based data of Miniworld."""

    def __init__(self, paths, config, transform, mode="ad"):
        config['store_gpu'] = False
        super().__init__(paths, config)

        assert mode in ["ad", "dpt"], "You must select mode from ad or dpt."

        self.transform = transform
        self.config = config
        self.mode = mode

        if self.mode == "dpt":
            context_filepaths = []
            query_images = []

            for traj in self.trajs:
                context_filepaths.append(traj['context_images'])
                query_image = self.transform(traj['query_image']).float()
                query_images.append(query_image)

            self.dataset.update({
                'context_filepaths': context_filepaths,
                'query_images': torch.stack(query_images),
            })
        
        else: # ad
            context_filepaths = []
            for traj in self.trajs:
                context_filepaths.append(traj['context_images'])
            
            self.dataset.update({
                'context_filepaths': context_filepaths,
            })

    def __getitem__(self, index):
        'Generates one sample of data'
        filepath = self.dataset['context_filepaths'][index]
        context_images = np.load(filepath)
        context_images = [self.transform(images) for images in context_images]
        context_images = torch.stack(context_images).float()

        if self.mode == "dpt":
            query_images = self.dataset['query_images'][index]
            res = {
                'context_images': context_images,#.to(device),
                'context_states': self.dataset['context_states'][index],
                'context_actions': self.dataset['context_actions'][index],
                'context_next_states': self.dataset['context_next_states'][index],
                'context_rewards': self.dataset['context_rewards'][index],
                'query_images': query_images,#.to(device),
                'query_states': self.dataset['query_states'][index],
                'optimal_actions': self.dataset['optimal_actions'][index],
                'zeros': self.zeros,
            }
        else:
            res = {
                'context_images': context_images,#.to(device),
                'context_states': self.dataset['context_states'][index],
                'context_actions': self.dataset['context_actions'][index],
                'context_next_states': self.dataset['context_next_states'][index],
                'context_rewards': self.dataset['context_rewards'][index],
                'zeros': self.zeros,
            }

        if self.shuffle:
            perm = torch.randperm(self.horizon)
            res['context_images'] = res['context_images'][perm]
            res['context_states'] = res['context_states'][perm]
            res['context_actions'] = res['context_actions'][perm]
            res['context_next_states'] = res['context_next_states'][perm]
            res['context_rewards'] = res['context_rewards'][perm]

        return res
    

class DarkroomDataset(torch.utils.data.Dataset):
    """"Dataset class for Darkroom."""

    def __init__(self, path, config, mode="ad", use_lambda=True):
        self.shuffle = config['shuffle']
        self.horizon = config['horizon']
        assert config['store_gpu'] == False
        self.store_gpu = config['store_gpu']
        self.config = config
        self.use_lambda = use_lambda
    
        assert mode in ["ad", "dpt"], "You must select mode from ad or dpt."
        self.config = config
        self.mode = mode
        traj = pickle.load(open(path, 'rb'))

        context_states = traj['observations']
        context_actions = traj['actions']
        context_rewards = traj['rewards']
        if use_lambda:
            context_lambda = traj['lambda']

        
        if len(context_rewards.shape) < 3:
            context_rewards = context_rewards[:, :, None]
        if len(context_actions.shape) < 3:
            context_actions = context_actions[:, :, None]
        if use_lambda:
            if len(context_lambda.shape) < 3:
                context_lambda = context_lambda[:, :, None]

        if use_lambda:
            self.dataset = {
                'context_states': convert_to_tensor(context_states, store_gpu=self.store_gpu),
                'context_actions': convert_to_tensor(context_actions, store_gpu=self.store_gpu),
                'context_rewards': convert_to_tensor(context_rewards, store_gpu=self.store_gpu),
                'context_lambda': convert_to_tensor(context_lambda, store_gpu=self.store_gpu),
            }
        else:
            self.dataset = {
                'context_states': convert_to_tensor(context_states, store_gpu=self.store_gpu),
                'context_actions': convert_to_tensor(context_actions, store_gpu=self.store_gpu),
                'context_rewards': convert_to_tensor(context_rewards, store_gpu=self.store_gpu),
            }

    def __len__(self):
        return len(self.dataset['context_states'])

    def __getitem__(self, index):
        'Generates one sample of data'

        if self.use_lambda:
            res = {
                'context_states': self.dataset['context_states'][index],
                'context_actions': self.dataset['context_actions'][index],
                'context_rewards': self.dataset['context_rewards'][index],
                'context_lambda': self.dataset['context_lambda'][index],
            }
        else:
            res = {
                'context_states': self.dataset['context_states'][index],
                'context_actions': self.dataset['context_actions'][index],
                'context_rewards': self.dataset['context_rewards'][index],
            }

        if self.shuffle:
            perm = torch.randperm(self.horizon)
            res['context_states'] = res['context_states'][perm]
            res['context_actions'] = res['context_actions'][perm]
            res['context_rewards'] = res['context_rewards'][perm]
            if self.mode == "dpt":
                res['context_next_states'] = res['context_next_states'][perm]
            if self.use_lambda:
                res['context_lambda'] = res['context_lambda'][perm]

        return res


class MinigridDataset(torch.utils.data.Dataset):
    """"Dataset class for Minigrid."""

    def __init__(self, path, config, mode="ad", include=[], reward_mode=None, use_reward=True, ):
        self.shuffle = config['shuffle']
        self.horizon = config['horizon']
        assert config['store_gpu'] == False
        self.store_gpu = config['store_gpu']
        self.config = config
        self.reward_mode = reward_mode

        logger.info("Dataset Init! reward_mode: {}".format(reward_mode))
    
        assert mode in ["ad", "dpt"], "You must select mode from ad or dpt."
        self.include = include
        self.config = config
        self.mode = mode
        if type(path) is not list:
            traj = pickle.load(open(path, 'rb'))
        else:
            trajs = []
            for p in path:
                with open(p, 'rb') as f:
                    traj = pickle.load(open(p, 'rb'))
                    trajs.append(traj)

        if type(path) is list:
            context_states = []
            context_actions = []
            context_rewards = []
            if "values" in include:
                context_values = []
            for traj in trajs:
                logger.info(f"Processing trajectory with {len(traj['observations'])} observations.")
                if traj['observations'].shape[0] > 80000:
                    max_len = 80000
                    context_states.append(traj['observations'][:max_len])
                    context_actions.append(traj['actions'][:max_len])
                    context_rewards.append(traj['rewards'][:max_len])
                    if "values" in include:
                        context_values.append(traj['values'][:max_len])
                    # free memory
                    del traj['observations'], traj['actions'], traj['rewards']
                    if "values" in include:
                        del traj['values']
                else:
                    context_states.append(traj['observations'])
                    context_actions.append(traj['actions'])
                    context_rewards.append(traj['rewards'])
                    if "values" in include:
                        context_values.append(traj['values'])

            # context_states = np.array(context_states)
            context_actions = np.concatenate(context_actions, axis=0)
            context_states = np.concatenate(context_states, axis=0)
            context_rewards = np.concatenate(context_rewards, axis=0)
            if "values" in include:
                context_values = np.concatenate(context_values, axis=0)

            logger.info(f"Context states shape: {context_states.shape}")
            logger.info(f"Context actions shape: {context_actions.shape}")
            logger.info(f"Context rewards shape: {context_rewards.shape}")
        else:
            context_states = traj['observations']
            context_actions = traj['actions']
            context_rewards = traj['rewards']
            if "values" in include:
                context_values = traj['values']

        
        if len(context_rewards.shape) < 3:
            context_rewards = context_rewards[:, :, None]
        if len(context_actions.shape) < 3:
            context_actions = context_actions[:, :, None]
        if "values" in include:
            if len(context_values.shape) < 3:
                context_values = context_values[:, :, None]

        if not use_reward:
            context_rewards = np.zeros_like(context_rewards)

        if "values" in include:
            self.dataset = {
                'context_states': convert_to_tensor(context_states, store_gpu=self.store_gpu),
                'context_actions': convert_to_tensor(context_actions, store_gpu=self.store_gpu),
                'context_rewards': convert_to_tensor(context_rewards, store_gpu=self.store_gpu),
                'context_values': convert_to_tensor(context_values, store_gpu=self.store_gpu),
            }
        else:
            self.dataset = {
                'context_states': convert_to_tensor(context_states, store_gpu=self.store_gpu),
                'context_actions': convert_to_tensor(context_actions, store_gpu=self.store_gpu),
                'context_rewards': convert_to_tensor(context_rewards, store_gpu=self.store_gpu),
            }

    def __len__(self):
        return len(self.dataset['context_states'])

    def __getitem__(self, index):
        'Generates one sample of data'

        if "values" in self.include:
            res = {
                'context_states': self.dataset['context_states'][index],
                'context_actions': self.dataset['context_actions'][index],
                'context_rewards': self.dataset['context_rewards'][index],
                'context_values': self.dataset['context_values'][index],
            }
        elif self.reward_mode is not None:
            context_states = self.dataset['context_states'][index]
            context_actions = self.dataset['context_actions'][index]
            context_rewards = self.dataset['context_rewards'][index]
            # calculate the reward signals
            if self.reward_mode == 0:
                ## compute the remaining reward, i.e. the sum of rewards from the current step to the end of the trajectory
                context_values = torch.zeros_like(context_rewards)
                for i in range(self.horizon):
                    context_values[i] = context_rewards[i:].sum()
            elif self.reward_mode == 1:
                ## compute the remaining average reward
                context_values = torch.zeros_like(context_rewards)
                for i in range(self.horizon):
                    context_values[i] = context_rewards[i:].mean() * self.horizon
            elif self.reward_mode == 2:
                ## compute the remaining average reward
                context_values = torch.zeros_like(context_rewards)
                context_values[0] = context_rewards.mean() * self.horizon
                for i in range(1, self.horizon):
                    context_values[i] = max(context_rewards[i:].mean() * self.horizon, context_values[i-1])
            res = {
                'context_states': context_states,
                'context_actions': context_actions,
                'context_rewards': context_rewards,
                'context_values': context_values,
            }
        else:
            res = {
                'context_states': self.dataset['context_states'][index],
                'context_actions': self.dataset['context_actions'][index],
                'context_rewards': self.dataset['context_rewards'][index],
            }

        if self.shuffle:
            perm = torch.randperm(self.horizon)
            res['context_states'] = res['context_states'][perm]
            res['context_actions'] = res['context_actions'][perm]
            res['context_rewards'] = res['context_rewards'][perm]
            if self.mode == "dpt":
                res['context_next_states'] = res['context_next_states'][perm]
            if "values" in self.include:
                res['context_values'] = res['context_values'][perm]

        return res