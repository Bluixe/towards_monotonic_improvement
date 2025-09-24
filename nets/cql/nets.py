
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
    
class ActorNet(nn.Module):
    def __init__(self, obs_space, action_space, device, args):
        super().__init__()
        self.device = device
        self.obs_shape = get_shape_from_obs_space(obs_space)
        self.action_shape = action_space.n

        self.cnn_layers_params = "32,3,1,1 64,3,1,1 32,3,1,1"
        self.cnn = CNNBase(obs_shape=self.obs_shape, args=args, cnn_layers_params=self.cnn_layers_params)
        input_size = self.cnn.hidden_size

        self.output_layer = nn.Linear(input_size, self.action_shape)

    def forward(self, obs, state=None, info=None):
        x = obs
        x = self.cnn(x)
        x = self.output_layer(x)
        return x, state
    
    def get_action(self, obs):
        logits, _ = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.detach().cpu()
    

class VValueNet(nn.Module):
    def __init__(self, obs_space, device, args):
        super().__init__()
        self.device = device
        self.obs_shape = get_shape_from_obs_space(obs_space)
        

        self.cnn_layers_params = "32,3,1,1 64,3,1,1 32,3,1,1"
        self.cnn = CNNBase(obs_shape=self.obs_shape, args=args, cnn_layers_params=self.cnn_layers_params)
        input_size = self.cnn.hidden_size

        self.output_layer = nn.Linear(input_size, 1)

    def forward(self, obs, state=None, info=None):
        x = obs
        x = self.cnn(x)
        x = self.output_layer(x)
        return x, state

    
class QValueNet(nn.Module):
    def __init__(self, obs_space, action_space, num_quantiles, device, args):
        super().__init__()
        self.device = device
        self.obs_shape = obs_space
        self.action_shape = action_space
        self.num_quantiles = num_quantiles
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(args["dropout"]),
            nn.Flatten(start_dim=1),
            nn.Linear(int(16 * self.obs_shape * self.obs_shape), args["n_embd"]),
            nn.ReLU(),
        )
        input_size = args["n_embd"]

        self.output_layer = nn.Linear(input_size, self.action_shape * self.num_quantiles)

    def forward(self, obs, state=None, info=None):
        x = obs
        x = self.cnn(x)
        x = self.output_layer(x)
        x = x.view(-1, self.action_shape, self.num_quantiles)
        return x, state
    