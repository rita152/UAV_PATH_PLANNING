"""
MASAC 智能体
包含 Actor、Critic 和 Entropy 调节器
"""
import torch
import torch.nn as nn
from .model import ActorNet, CriticNet


class Actor:
    def __init__(self, state_dim, action_dim, max_action, min_action, hidden_dim, policy_lr, 
                 device='cpu', use_layer_norm=True):
        self.max_action = max_action
        self.min_action = min_action
        self.device = torch.device(device)
        
        # 创建网络并移到设备（支持 Layer Normalization）
        self.action_net = ActorNet(state_dim, action_dim, max_action, hidden_dim, 
                                   use_layer_norm=use_layer_norm)
        self.action_net.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.action_net.parameters(), lr=policy_lr)

    @torch.no_grad()
    def choose_action(self, state):
        """
        选择动作（用于训练时的环境交互）- 输入CPU numpy，输出CPU numpy
        
        使用随机策略进行探索
        
        Args:
            state: 输入状态 (CPU numpy)
            
        Returns:
            action: 采样的动作 (CPU numpy)
        """
        # CPU numpy → GPU tensor
        state_tensor = torch.FloatTensor(state).to(self.device)
        mean, std = self.action_net(state_tensor)
        distribution = torch.distributions.Normal(mean, std)
        action = distribution.sample()
        action = torch.clamp(action, self.min_action, self.max_action)
        # GPU tensor → CPU numpy
        return action.cpu().numpy()
    
    @torch.no_grad()
    def choose_action_deterministic(self, state):
        """
        确定性动作选择（用于测试/评估）- 输入CPU numpy，输出CPU numpy
        
        直接使用均值，不进行随机采样，获得稳定的测试结果
        
        Args:
            state: 输入状态 (CPU numpy)
            
        Returns:
            action: 确定性动作 (CPU numpy)
        """
        # CPU numpy → GPU tensor
        state_tensor = torch.FloatTensor(state).to(self.device)
        mean, _ = self.action_net(state_tensor)  # 忽略 std
        action = torch.clamp(mean, self.min_action, self.max_action)
        # GPU tensor → CPU numpy
        return action.cpu().numpy()
    
    @staticmethod
    @torch.no_grad()
    def choose_actions_batch(actors, states, device):
        """
        批量选择动作（优化CPU-GPU传输）- 输入CPU numpy数组，输出CPU numpy数组
        
        一次性处理所有agent的动作选择，减少CPU-GPU数据传输次数
        
        Args:
            actors: Actor列表 [n_agents]
            states: 所有agent的状态 [n_agents, state_dim] (CPU numpy)
            device: 计算设备
            
        Returns:
            actions: 所有agent的动作 [n_agents, action_dim] (CPU numpy)
        """
        n_agents = len(actors)
        actions = []
        
        # 一次性将所有状态转移到GPU
        states_tensor = torch.FloatTensor(states).to(device)  # [n_agents, state_dim]
        
        # 批量计算所有agent的动作
        for i in range(n_agents):
            mean, std = actors[i].action_net(states_tensor[i])
            distribution = torch.distributions.Normal(mean, std)
            action = distribution.sample()
            action = torch.clamp(action, actors[i].min_action, actors[i].max_action)
            actions.append(action)
        
        # 拼接并一次性转回CPU
        actions_tensor = torch.stack(actions, dim=0)  # [n_agents, action_dim]
        return actions_tensor.cpu().numpy()
    
    @staticmethod
    @torch.no_grad()
    def choose_actions_batch_deterministic(actors, states, device):
        """
        批量确定性动作选择（优化CPU-GPU传输）- 输入CPU numpy数组，输出CPU numpy数组
        
        用于测试时的批量动作选择，使用确定性策略（均值）
        
        Args:
            actors: Actor列表 [n_agents]
            states: 所有agent的状态 [n_agents, state_dim] (CPU numpy)
            device: 计算设备
            
        Returns:
            actions: 所有agent的动作 [n_agents, action_dim] (CPU numpy)
        """
        n_agents = len(actors)
        actions = []
        
        # 一次性将所有状态转移到GPU
        states_tensor = torch.FloatTensor(states).to(device)  # [n_agents, state_dim]
        
        # 批量计算所有agent的动作（使用均值）
        for i in range(n_agents):
            mean, _ = actors[i].action_net(states_tensor[i])
            action = torch.clamp(mean, actors[i].min_action, actors[i].max_action)
            actions.append(action)
        
        # 拼接并一次性转回CPU
        actions_tensor = torch.stack(actions, dim=0)  # [n_agents, action_dim]
        return actions_tensor.cpu().numpy()
    
    def evaluate(self, state):
        """
        评估动作（用于训练）- 输入输出都是GPU tensor
        
        使用正确的重参数化技巧（reparameterization trick）:
        1. 使用 rsample() 保持梯度
        2. 对 tanh 变换前的值计算 log_prob
        3. 应用 tanh 修正
        4. 对动作维度求和
        
        Args:
            state: 输入状态 (GPU tensor)
            
        Returns:
            action: 采样的动作 (GPU tensor)
            log_prob: 对数概率 (GPU tensor)
        """
        # state 已经是 GPU 上的 tensor
        mean, std = self.action_net(state)
        normal = torch.distributions.Normal(mean, std)
        
        # 重参数化采样（保持梯度）
        x_t = normal.rsample()  # ✅ 使用 rsample 而不是 sample
        action = torch.tanh(x_t)
        action = torch.clamp(action, self.min_action, self.max_action)
        
        # 计算 log_prob 并应用 tanh 修正
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # ✅ 对动作维度求和
        
        return action, log_prob

    def update(self, loss):
        """
        更新Actor网络参数
        
        Args:
            loss: Actor损失
        """
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.action_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()  # 返回loss值用于日志记录

class Entropy:
    """
    熵调节器（Temperature Tuning）
    自动调节SAC算法中的熵系数
    """
    def __init__(self, target_entropy, lr, device='cpu'):
        self.target_entropy = target_entropy
        self.device = torch.device(device)
        
        # log_alpha需要在指定设备上
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    def update(self, loss):
        """
        更新熵系数
        
        Args:
            loss: 熵损失
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.alpha = self.log_alpha.exp()
        
        return loss.item()  # 返回loss值用于日志记录

class Critic:
    """
    Critic 网络（Q值估计）
    使用 Double Q-Network 减少过估计
    """
    def __init__(self, state_dim, action_dim, hidden_dim, value_lr, tau, device='cpu', use_layer_norm=True):
        self.tau = tau
        self.device = torch.device(device)
        
        # 创建网络并移到设备（支持 Layer Normalization）
        self.critic_net = CriticNet(state_dim, action_dim, hidden_dim, use_layer_norm=use_layer_norm)
        self.target_critic_net = CriticNet(state_dim, action_dim, hidden_dim, use_layer_norm=use_layer_norm)
        self.critic_net.to(self.device)
        self.target_critic_net.to(self.device)
        
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())
        self.optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=value_lr, eps=1e-5)
        self.loss_fn = nn.MSELoss()
    
    def soft_update(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def get_q_value(self, state, action):
        """
        获取当前Q值
        
        Args:
            state: 状态
            action: 动作
            
        Returns:
            q1, q2: 两个Q值
        """
        return self.critic_net(state, action)

    @torch.no_grad()
    def get_target_q_value(self, state, action):
        """
        获取目标Q值（无梯度计算）
        
        Args:
            state: 状态
            action: 动作
            
        Returns:
            q1, q2: 两个目标Q值
        """
        return self.target_critic_net(state, action)

    def update(self, q1_current, q2_current, q_target):
        """
        更新Critic网络
        
        Args:
            q1_current: 当前Q1值
            q2_current: 当前Q2值
            q_target: 目标Q值
        """
        loss = self.loss_fn(q1_current, q_target) + self.loss_fn(q2_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()  # 返回loss值用于日志记录