"""
Actor-Critic 神经网络模型
包含 ActorNet 和 CriticNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):
    """
    Actor 神经网络
    输入状态，输出动作的均值和标准差
    """
    def __init__(self, state_dim, action_dim, max_action, hidden_dim):
        super(ActorNet, self).__init__()
        self.max_action = max_action
        
        # 网络层
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc1.weight.data.normal_(0, 0.1)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2.weight.data.normal_(0, 0.1)
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.mean_layer.weight.data.normal_(0, 0.1)
        
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer.weight.data.normal_(0, 0.1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.max_action * torch.tanh(self.mean_layer(x))
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        
        return mean, std

class CriticNet(nn.Module):
    """
    Critic 神经网络（Double Q-Network）
    输入状态和动作，输出两个Q值
    """
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(CriticNet, self).__init__()
        
        # Q1 网络
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc1.weight.data.normal_(0, 0.1)
        
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc2.weight.data.normal_(0, 0.1)
        
        self.q1_out = nn.Linear(hidden_dim, 1)
        self.q1_out.weight.data.normal_(0, 0.1)
        
        # Q2 网络
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc1.weight.data.normal_(0, 0.1)
        
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc2.weight.data.normal_(0, 0.1)
        
        self.q2_out = nn.Linear(hidden_dim, 1)
        self.q2_out.weight.data.normal_(0, 0.1)
    
    def forward(self, state, action):
        state_action = torch.cat((state, action), dim=1)
        
        # Q1 前向传播
        q1 = F.relu(self.q1_fc1(state_action))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)
        
        # Q2 前向传播
        q2 = F.relu(self.q2_fc1(state_action))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)
        
        return q1, q2