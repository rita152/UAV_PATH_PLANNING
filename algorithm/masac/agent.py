"""
MASAC 智能体
包含 Actor、Critic 和 Entropy 调节器
"""
import torch
import torch.nn as nn
from .model import ActorNet, CriticNet


class Actor:
    def __init__(self, state_dim, action_dim, max_action, min_action, hidden_dim, policy_lr):
        self.max_action = max_action
        self.min_action = min_action
        self.action_net = ActorNet(state_dim, action_dim, max_action, hidden_dim)
        self.optimizer = torch.optim.Adam(self.action_net.parameters(), lr=policy_lr)

    def choose_action(self, state):
        """选择动作（用于环境交互）"""
        state_tensor = torch.FloatTensor(state)
        mean, std = self.action_net(state_tensor)
        distribution = torch.distributions.Normal(mean, std)
        action = distribution.sample()
        action = torch.clamp(action, self.min_action, self.max_action)
        return action.detach().numpy()
    
    def evaluate(self, state):
        """评估动作（用于训练，返回动作和对数概率）"""
        state_tensor = torch.FloatTensor(state)
        mean, std = self.action_net(state_tensor)
        distribution = torch.distributions.Normal(mean, std)
        
        noise_distribution = torch.distributions.Normal(0, 1)
        noise = noise_distribution.sample()
        
        action = torch.tanh(mean + std * noise)
        action = torch.clamp(action, self.min_action, self.max_action)
        
        log_prob = distribution.log_prob(mean + std * noise) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob

    def update(self, loss):
        """更新网络参数"""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Entropy:
    """
    熵调节器（Temperature Tuning）
    自动调节SAC算法中的熵系数
    """
    def __init__(self, target_entropy, lr):
        self.target_entropy = target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp()
        self.optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    def update(self, loss):
        """更新熵系数"""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.alpha = self.log_alpha.exp()

class Critic:
    """
    Critic 网络（Q值估计）
    使用 Double Q-Network 减少过估计
    """
    def __init__(self, state_dim, action_dim, hidden_dim, value_lr, tau):
        self.tau = tau
        self.critic_net = CriticNet(state_dim, action_dim, hidden_dim)
        self.target_critic_net = CriticNet(state_dim, action_dim, hidden_dim)
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())
        self.optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=value_lr, eps=1e-5)
        self.loss_fn = nn.MSELoss()
    
    def soft_update(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def get_q_value(self, state, action):
        """获取Q值"""
        return self.critic_net(state, action)

    def get_target_q_value(self, state, action):
        """获取目标Q值"""
        return self.target_critic_net(state, action)

    def update(self, q1_current, q2_current, q_target):
        """更新Critic网络"""
        loss = self.loss_fn(q1_current, q_target) + self.loss_fn(q2_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()