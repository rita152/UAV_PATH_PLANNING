import numpy as np


class Memory:
    """经验回放缓冲区"""
    
    def __init__(self, capacity, dims):
        """
        初始化记忆库
        
        Args:
            capacity: 记忆库容量
            dims: 每条记忆的维度(state_dim + action_dim + reward_dim + next_state_dim)
        """
        self.capacity = capacity
        self.mem = np.zeros((capacity, dims))
        self.memory_counter = 0
    
    def store_transition(self, s, a, r, s_):
        """
        存储一条转移经验
        
        Args:
            s: 当前状态
            a: 动作
            r: 奖励
            s_: 下一状态
        """
        tran = np.hstack((s, a, r, s_))
        index = self.memory_counter % self.capacity
        self.mem[index, :] = tran
        self.memory_counter += 1
    
    def sample(self, n):
        """
        从记忆库中随机采样
        
        Args:
            n: 采样数量
            
        Returns:
            采样的经验batch
        """
        assert self.memory_counter >= self.capacity, '记忆库没有存满记忆'
        sample_index = np.random.choice(self.capacity, n)
        new_mem = self.mem[sample_index, :]
        return new_mem
    
    def is_ready(self):
        """检查记忆库是否已满，可以开始采样"""
        return self.memory_counter >= self.capacity