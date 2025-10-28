"""
MASAC (Multi-Agent Soft Actor-Critic) 算法模块

包含：
- Actor: 策略网络（带确定性动作选择）
- Critic: 价值网络（带梯度裁剪）
- Entropy: 熵调节器
- Memory: 优先级经验回放缓冲区（PER）
- ActorNet: Actor 神经网络（He初始化 + Layer Normalization）
- CriticNet: Critic 神经网络（He初始化 + Layer Normalization）
- Trainer: 训练器
- Tester: 测试器
"""

from .agent import Actor, Critic, Entropy
from .buffer import Memory
from .model import ActorNet, CriticNet
from .trainer import Trainer
from .tester import Tester

__all__ = [
    'Actor',
    'Critic',
    'Entropy',
    'Memory',
    'ActorNet',
    'CriticNet',
    'Trainer',
    'Tester'
]

__version__ = '1.0.0'

