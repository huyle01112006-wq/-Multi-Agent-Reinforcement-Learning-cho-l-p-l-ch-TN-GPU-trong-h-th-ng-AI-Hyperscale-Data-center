# agents.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim=2):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size=32):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)
        return (np.array(obs), np.array(action), np.array(reward), 
                np.array(next_obs), np.array(done))

    def __len__(self):
        return len(self.buffer)

class MARLAgents:
    def __init__(self, num_nodes, obs_dim, action_dim=2, lr=0.001, gamma=0.95):
        self.num_nodes = num_nodes
        self.gamma = gamma
        
        # Policy Network & Target Network cho ổn định DQN
        self.q_networks = [QNetwork(obs_dim, action_dim) for _ in range(num_nodes)]
        self.target_networks = [QNetwork(obs_dim, action_dim) for _ in range(num_nodes)]
        
        # Đồng bộ trọng số ban đầu
        for i in range(num_nodes):
            self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())

        self.optimizers = [optim.Adam(q_net.parameters(), lr=lr) for q_net in self.q_networks]
        self.buffer = ReplayBuffer()

    def select_actions(self, obs_batch, epsilon):
        actions = []
        for i in range(self.num_nodes):
            if random.random() < epsilon:
                actions.append(random.choice([0, 1]))
            else:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs_batch[i])
                    q_vals = self.q_networks[i](obs_t)
                    actions.append(torch.argmax(q_vals).item())
        return actions

    def update_target_network(self):
        """Được gọi định kỳ để giúp DQN hội tụ ổn định"""
        for i in range(self.num_nodes):
            self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())

    def train_step(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return

        b_obs, b_actions, b_rewards, b_next_obs, b_dones = self.buffer.sample(batch_size)
        
        for i in range(self.num_nodes):
            node_obs = torch.FloatTensor(b_obs[:, i])
            node_act = torch.LongTensor(b_actions[:, i]).unsqueeze(1)
            node_rew = torch.FloatTensor(b_rewards[:, i]).unsqueeze(1) # Credit Assignment theo Node
            node_next_obs = torch.FloatTensor(b_next_obs[:, i])
            node_done = torch.FloatTensor(b_dones).unsqueeze(1)

            # DQN Loss với Target Network
            q_val = self.q_networks[i](node_obs).gather(1, node_act)
            with torch.no_grad():
                next_q = self.target_networks[i](node_next_obs).max(1)[0].unsqueeze(1)
                target = node_rew + (1 - node_done) * self.gamma * next_q

            loss = nn.MSELoss()(q_val, target)
            self.optimizers[i].zero_grad()
            loss.backward()
            self.optimizers[i].step()