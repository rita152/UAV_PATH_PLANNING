"""
Ornstein-Uhlenbeck 噪声生成器
用于SAC算法的探索噪声
"""
import numpy as np


class Ornstein_Uhlenbeck_Noise:
    """
    Ornstein-Uhlenbeck 过程噪声生成器
    
    用于连续动作空间的探索，生成时间相关的噪声
    公式：dXt = θ(μ - Xt)dt + σ√dt·εt
    其中 θ 是回归速度，μ 是均值，σ 是波动率
    """
    def __init__(self, mean, sigma, theta, dt, initial_noise=None):
        self.mean = mean
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.initial_noise = initial_noise
        self.reset()

    def __call__(self):
        """生成下一个噪声值"""
        # dXt = θ(μ - Xt)dt + σ√dt·εt
        drift = self.theta * (self.mean - self.current_noise) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        self.current_noise = self.current_noise + drift + diffusion
        return self.current_noise

    def reset(self):
        """重置噪声状态"""
        if self.initial_noise is not None:
            self.current_noise = self.initial_noise
        else:
            self.current_noise = np.zeros_like(self.mean)