"""
Actor-Critic 神经网络模型
包含 ActorNet 和 CriticNet

改进点：
1. 使用 He/Xavier 初始化替代 Normal(0, 0.1)
2. 添加 Layer Normalization 提高训练稳定性
3. 支持可选的更深网络结构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):
    """
    Actor 神经网络（改进版）
    
    输入状态，输出动作的均值和标准差
    
    改进：
    - 使用 He 初始化（适合 ReLU 激活函数）
    - 添加 Layer Normalization 稳定训练
    - 支持更深的网络结构（可选）
    """
    def __init__(self, state_dim, action_dim, max_action, hidden_dim, use_layer_norm=True):
        super(ActorNet, self).__init__()
        self.max_action = max_action
        self.use_layer_norm = use_layer_norm
        
        # 第一层
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self._init_weights(self.fc1)
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(hidden_dim)
        
        # 第二层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self._init_weights(self.fc2)
        if use_layer_norm:
            self.ln2 = nn.LayerNorm(hidden_dim)
        
        # 输出层（均值和标准差）
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self._init_weights(self.mean_layer, std=0.01)  # 输出层使用较小的初始化
        
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self._init_weights(self.log_std_layer, std=0.01)
    
    def _init_weights(self, layer, std=None):
        """
        使用 He 初始化（适合 ReLU）
        
        Args:
            layer: 要初始化的层
            std: 可选的标准差（用于输出层）
        """
        if std is not None:
            # 对于输出层，使用较小的标准差
            nn.init.normal_(layer.weight, mean=0.0, std=std)
        else:
            # He 初始化（适合 ReLU）
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

    def forward(self, state):
        """
        前向传播
        
        Args:
            state: 输入状态 [batch, state_dim]
            
        Returns:
            mean: 动作均值 [batch, action_dim]
            std: 动作标准差 [batch, action_dim]
        """
        # 第一层 + LayerNorm + ReLU
        x = self.fc1(state)
        if self.use_layer_norm:
            x = self.ln1(x)
        x = F.relu(x)
        
        # 第二层 + LayerNorm + ReLU
        x = self.fc2(x)
        if self.use_layer_norm:
            x = self.ln2(x)
        x = F.relu(x)
        
        # 输出层
        mean = self.max_action * torch.tanh(self.mean_layer(x))
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)  # 防止数值不稳定
        std = log_std.exp()
        
        return mean, std

class CriticNet(nn.Module):
    """
    Critic 神经网络（Double Q-Network，改进版）
    
    输入状态和动作，输出两个Q值
    
    改进：
    - 使用 He 初始化（适合 ReLU 激活函数）
    - 添加 Layer Normalization 稳定训练
    - 独立的两个 Q 网络减少过估计
    """
    def __init__(self, state_dim, action_dim, hidden_dim, use_layer_norm=True):
        super(CriticNet, self).__init__()
        self.use_layer_norm = use_layer_norm
        
        # Q1 网络
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self._init_weights(self.q1_fc1)
        if use_layer_norm:
            self.q1_ln1 = nn.LayerNorm(hidden_dim)
        
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self._init_weights(self.q1_fc2)
        if use_layer_norm:
            self.q1_ln2 = nn.LayerNorm(hidden_dim)
        
        self.q1_out = nn.Linear(hidden_dim, 1)
        self._init_weights(self.q1_out, std=0.01)
        
        # Q2 网络
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self._init_weights(self.q2_fc1)
        if use_layer_norm:
            self.q2_ln1 = nn.LayerNorm(hidden_dim)
        
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self._init_weights(self.q2_fc2)
        if use_layer_norm:
            self.q2_ln2 = nn.LayerNorm(hidden_dim)
        
        self.q2_out = nn.Linear(hidden_dim, 1)
        self._init_weights(self.q2_out, std=0.01)
    
    def _init_weights(self, layer, std=None):
        """
        使用 He 初始化（适合 ReLU）
        
        Args:
            layer: 要初始化的层
            std: 可选的标准差（用于输出层）
        """
        if std is not None:
            # 对于输出层，使用较小的标准差
            nn.init.normal_(layer.weight, mean=0.0, std=std)
        else:
            # He 初始化（适合 ReLU）
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, state, action):
        """
        前向传播
        
        Args:
            state: 输入状态 [batch, state_dim]
            action: 输入动作 [batch, action_dim]
            
        Returns:
            q1: 第一个Q值 [batch, 1]
            q2: 第二个Q值 [batch, 1]
        """
        state_action = torch.cat((state, action), dim=1)
        
        # Q1 前向传播
        q1 = self.q1_fc1(state_action)
        if self.use_layer_norm:
            q1 = self.q1_ln1(q1)
        q1 = F.relu(q1)
        
        q1 = self.q1_fc2(q1)
        if self.use_layer_norm:
            q1 = self.q1_ln2(q1)
        q1 = F.relu(q1)
        
        q1 = self.q1_out(q1)
        
        # Q2 前向传播
        q2 = self.q2_fc1(state_action)
        if self.use_layer_norm:
            q2 = self.q2_ln1(q2)
        q2 = F.relu(q2)
        
        q2 = self.q2_fc2(q2)
        if self.use_layer_norm:
            q2 = self.q2_ln2(q2)
        q2 = F.relu(q2)
        
        q2 = self.q2_out(q2)
        
        return q1, q2