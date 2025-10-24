import numpy as np


class Ornstein_Uhlenbeck_Noise:
    """Ornstein-Uhlenbeck噪声过程，用于探索"""
    
    def __init__(self, mu, sigma=0.1, theta=0.1, dt=1e-2, x0=None):
        """
        初始化OU噪声
        
        Args:
            mu: 均值
            sigma: 噪声标准差
            theta: 回归系数
            dt: 时间步长
            x0: 初始状态
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        """生成下一个噪声值"""
        x = self.x_prev + \
            self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        """重置噪声状态"""
        if self.x0 is not None:
            self.x_prev = self.x0
        else:
            self.x_prev = np.zeros_like(self.mu)