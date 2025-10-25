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
            
        Raises:
            ValueError: 当记忆库样本不足时
        """
        # ✅ 修复：只要有足够的样本就可以采样，不需要等buffer填满
        current_size = min(self.memory_counter, self.capacity)
        
        # ✅ 修复：使用显式异常而非断言（断言可能被优化掉）
        if current_size < n:
            raise ValueError(
                f'记忆库样本不足: 需要{n}个样本进行采样，当前只有{current_size}个样本。\n'
                f'建议：\n'
                f'  1. 等待收集更多经验（至少{n}个）后再开始训练\n'
                f'  2. 或者减小batch_size参数'
            )
        
        # 从已有样本中随机采样（不重复采样）
        sample_index = np.random.choice(current_size, n, replace=False)
        new_mem = self.mem[sample_index, :]
        return new_mem
    
    def is_ready(self, batch_size):
        """
        检查记忆库是否有足够样本可以采样
        
        Args:
            batch_size: 需要采样的批次大小
            
        Returns:
            bool: 是否有足够样本进行采样
        """
        # ✅ 修复：与sample()逻辑一致，只需要>=batch_size即可
        current_size = min(self.memory_counter, self.capacity)
        return current_size >= batch_size