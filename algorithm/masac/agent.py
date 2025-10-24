import torch
import torch.nn as nn
from .model import ActorNet, CriticNet


class Actor:
    """Actor智能体，负责选择动作"""
    
    def __init__(self, state_dim, action_dim, max_action, min_action, lr=1e-3):
        """
        初始化Actor
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            max_action: 最大动作值
            min_action: 最小动作值
            lr: 学习率
        """
        self.max_action = max_action
        self.min_action = min_action
        self.action_net = ActorNet(state_dim, action_dim, max_action)
        self.optimizer = torch.optim.Adam(self.action_net.parameters(), lr=lr)

    def choose_action(self, s):
        """
        选择动作(用于训练)
        
        Args:
            s: 当前状态
            
        Returns:
            action: 选择的动作
        """
        inputstate = torch.FloatTensor(s)
        mean, std = self.action_net(inputstate)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, self.min_action, self.max_action)
        return action.detach().numpy()
    
    def evaluate(self, s):
        """
        评估状态并返回动作和对数概率(用于策略更新)
        
        Args:
            s: 状态
            
        Returns:
            action: 动作
            action_logprob: 动作对数概率
        """
        inputstate = torch.FloatTensor(s)
        mean, std = self.action_net(inputstate)
        dist = torch.distributions.Normal(mean, std)
        noise = torch.distributions.Normal(0, 1)
        z = noise.sample()
        action = torch.tanh(mean + std * z)
        action = torch.clamp(action, self.min_action, self.max_action)
        action_logprob = dist.log_prob(mean + std * z) - torch.log(1 - action.pow(2) + 1e-6)
        # 对所有动作维度求和，得到总的log_prob
        action_logprob = action_logprob.sum(dim=-1, keepdim=True)
        return action, action_logprob

    def learn(self, actor_loss):
        """
        更新Actor网络
        
        Args:
            actor_loss: Actor损失
        """
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()


class Entropy:
    """熵温度参数自动调节"""
    
    def __init__(self, target_entropy=-0.1, lr=3e-4):
        """
        初始化熵调节器
        
        Args:
            target_entropy: 目标熵值
            lr: 学习率
        """
        self.target_entropy = target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp()
        self.optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    def learn(self, entropy_loss):
        """
        更新熵温度参数
        
        Args:
            entropy_loss: 熵损失
        """
        self.optimizer.zero_grad()
        entropy_loss.backward()
        self.optimizer.step()


class Critic:
    """Critic智能体，负责评估状态-动作对的价值"""
    
    def __init__(self, state_dim, action_dim, lr=3e-3, tau=1e-2):
        """
        初始化Critic
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            lr: 学习率
            tau: 软更新系数
        """
        self.tau = tau
        self.critic_v = CriticNet(state_dim, action_dim)
        self.target_critic_v = CriticNet(state_dim, action_dim)
        self.target_critic_v.load_state_dict(self.critic_v.state_dict())
        self.optimizer = torch.optim.Adam(self.critic_v.parameters(), lr=lr, eps=1e-5)
        self.lossfunc = nn.MSELoss()
    
    def soft_update(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_critic_v.parameters(), self.critic_v.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def get_v(self, s, a):
        """
        获取Q值
        
        Args:
            s: 状态
            a: 动作
            
        Returns:
            q1, q2: 两个Q值
        """
        return self.critic_v(s, a)

    def target_get_v(self, s, a):
        """
        获取目标Q值
        
        Args:
            s: 状态
            a: 动作
            
        Returns:
            q1, q2: 两个目标Q值
        """
        return self.target_critic_v(s, a)

    def learn(self, current_q1, current_q2, target_q):
        """
        更新Critic网络
        
        Args:
            current_q1: 当前Q1值
            current_q2: 当前Q2值
            target_q: 目标Q值
        """
        loss = self.lossfunc(current_q1, target_q) + self.lossfunc(current_q2, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()