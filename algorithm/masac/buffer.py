"""
经验回放缓冲区（Replay Buffer）
用于存储和采样经验数据
"""
import numpy as np


class Memory:
    """
    经验回放缓冲区（Replay Buffer）
    存储和采样 (state, action, reward, next_state) 转换
    """
    def __init__(self, capacity, transition_dim):
        self.capacity = capacity
        self.buffer = np.zeros((capacity, transition_dim))
        self.counter = 0
    
    def store(self, state, action, reward, next_state):
        """存储一个转换"""
        transition = np.hstack((state, action, reward, next_state))
        index = self.counter % self.capacity
        self.buffer[index, :] = transition
        self.counter += 1
    
    def sample(self, batch_size):
        """随机采样一批转换"""
        assert self.counter >= self.capacity, '记忆库未满，无法采样'
        indices = np.random.choice(self.capacity, batch_size)
        batch = self.buffer[indices, :]
        return batch
    
    def is_ready(self, batch_size):
        """检查缓冲区是否有足够数据"""
        return self.counter >= batch_size