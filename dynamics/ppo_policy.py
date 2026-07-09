"""Minimal PPO policy wrapper for generating training trajectories."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class PPO_PolicyNet(nn.Module):
    """Simple policy network matching hrl-fc's architecture."""
    def __init__(self, input_dim=512, hidden_dim=128, action_dim=2, action_range=3.0):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.ln2 = nn.LayerNorm(hidden_dim * 2)
        self.linear3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.action_range = action_range
        self.action_dim = action_dim

    def forward(self, state):
        x = F.relu(self.ln1(self.linear1(state)))
        x = F.relu(self.ln2(self.linear2(x)))
        x = F.relu(self.ln3(self.linear3(x)))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, -3, 1)
        return mean, log_std

    @torch.no_grad()
    def get_action(self, state_np, noise_scale=0.1):
        """state_np: (512,) numpy array -> action: (2,) numpy array"""
        state = torch.FloatTensor(state_np).unsqueeze(0)
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        # Add noise for diversity
        z = mean + torch.randn_like(mean) * std * noise_scale
        action = torch.tanh(z) * self.action_range
        return action.numpy()[0]

class PPOGuide:
    """Load pretrained PPO policy from hrl-fc and provide action generation."""
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.policy = PPO_PolicyNet().to(device)
        policy_path = os.path.join(model_path, "PPO_policy.pth")
        self.policy.load_state_dict(torch.load(policy_path, map_location=device))
        self.policy.eval()
        self.max_speed = 15.0  # from config

    def get_action(self, wind_x_vec, wind_y_vec, as_state, target_pos, noise_scale=0.1):
        """Construct PPO state from components and return action.
        wind_x_vec: (252,) flattened compressed wind x-vector
        wind_y_vec: (252,) flattened compressed wind y-vector
        as_state: (6,) normalized state [x, y, vx, vy, time, energy]
        target_pos: (2,) pixel target position
        """
        wind_state = np.concatenate([
            wind_x_vec / self.max_speed,
            wind_y_vec / self.max_speed
        ])  # (504,)
        ppo_state = np.concatenate([wind_state, as_state, target_pos])  # (512,)
        return self.policy.get_action(ppo_state, noise_scale)
