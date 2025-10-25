import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):
    """Actor网络，输出动作的均值和标准差"""
    
    def __init__(self, inp, outp, max_action=1.0, hidden_dim=256):
        """
        初始化Actor网络
        
        Args:
            inp: 输入维度(状态维度)
            outp: 输出维度(动作维度)
            max_action: 最大动作值
            hidden_dim: 隐藏层维度
        """
        super(ActorNet, self).__init__()
        self.max_action = max_action
        
        self.in_to_y1 = nn.Linear(inp, hidden_dim)
        self.in_to_y1.weight.data.normal_(0, 0.1)
        
        self.y1_to_y2 = nn.Linear(hidden_dim, hidden_dim)
        self.y1_to_y2.weight.data.normal_(0, 0.1)
        
        self.out = nn.Linear(hidden_dim, outp)
        self.out.weight.data.normal_(0, 0.1)
        
        self.std_out = nn.Linear(hidden_dim, outp)
        self.std_out.weight.data.normal_(0, 0.1)

    def forward(self, inputstate):
        """
        前向传播
        
        Args:
            inputstate: 输入状态
            
        Returns:
            mean: 动作均值（原始空间，未经tanh变换）
            std: 动作标准差
        """
        x = self.in_to_y1(inputstate)
        x = F.relu(x)
        x = self.y1_to_y2(x)
        x = F.relu(x)
        
        # ✅ 修复：mean不做tanh变换，保持在原始空间
        # tanh变换将在采样时进行，确保只变换一次
        mean = self.out(x)
        
        log_std = self.std_out(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        
        return mean, std


class CriticNet(nn.Module):
    """Critic网络，使用双Q网络"""
    
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        """
        初始化Critic网络
        
        Args:
            input_dim: 状态输入维度
            output_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(CriticNet, self).__init__()
        
        # Q1网络
        self.in_to_y1 = nn.Linear(input_dim + output_dim, hidden_dim)
        self.in_to_y1.weight.data.normal_(0, 0.1)
        
        self.y1_to_y2 = nn.Linear(hidden_dim, hidden_dim)
        self.y1_to_y2.weight.data.normal_(0, 0.1)
        
        self.out = nn.Linear(hidden_dim, 1)
        self.out.weight.data.normal_(0, 0.1)
        
        # Q2网络
        self.q2_in_to_y1 = nn.Linear(input_dim + output_dim, hidden_dim)
        self.q2_in_to_y1.weight.data.normal_(0, 0.1)
        
        self.q2_y1_to_y2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_y1_to_y2.weight.data.normal_(0, 0.1)
        
        self.q2_out = nn.Linear(hidden_dim, 1)
        self.q2_out.weight.data.normal_(0, 0.1)
    
    def forward(self, s, a):
        """
        前向传播
        
        Args:
            s: 状态
            a: 动作
            
        Returns:
            q1: 第一个Q值
            q2: 第二个Q值
        """
        inputstate = torch.cat((s, a), dim=1)
        
        # Q1
        q1 = self.in_to_y1(inputstate)
        q1 = F.relu(q1)
        q1 = self.y1_to_y2(q1)
        q1 = F.relu(q1)
        q1 = self.out(q1)
        
        # Q2
        q2 = self.q2_in_to_y1(inputstate)
        q2 = F.relu(q2)
        q2 = self.q2_y1_to_y2(q2)
        q2 = F.relu(q2)
        q2 = self.q2_out(q2)
        
        return q1, q2