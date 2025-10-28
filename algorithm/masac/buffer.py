"""
优先级经验回放缓冲区（Prioritized Experience Replay Buffer）

基于论文: Schaul et al. "Prioritized Experience Replay" (2015)
链接: https://arxiv.org/abs/1511.05952

核心特性：
1. 优先级采样：基于TD-error分配采样概率
2. 重要性采样权重：修正采样偏差
3. float32内存优化：节省50%内存
4. 早期训练：只需batch_size个样本即可开始训练

使用方法：
    # 创建缓冲区
    memory = Memory(capacity=10000, transition_dim=32)
    
    # 存储经验
    memory.store(state, action, reward, next_state)
    
    # 采样经验（返回batch, weights, indices）
    batch, weights, indices = memory.sample(batch_size=64)
    
    # 计算TD-error并更新优先级
    td_errors = compute_td_errors(batch)
    memory.update_priorities(indices, td_errors)
"""
import numpy as np


class Memory:
    """
    优先级经验回放缓冲区（完全使用PER，不支持均匀采样）
    
    优先学习重要的经验，提高样本效率和收敛速度
    
    算法原理：
    - 采样概率: P(i) = p_i^α / Σ_k p_k^α
    - 重要性权重: w_i = (N * P(i))^(-β) / max_w
    - 优先级: p_i = |TD_error| + ε
    
    参数：
        capacity: 缓冲区最大容量
        transition_dim: 转换的维度 (state_dim*n + action_dim*n + reward_dim*n + next_state_dim*n)
        alpha: 优先级指数 (0=均匀, 1=完全优先级)，推荐0.6
        beta: 重要性采样权重指数，从初始值逐渐增长到1.0
        beta_increment: beta的增长速率，推荐0.001
        epsilon: 防止优先级为0的小常数
    """
    
    def __init__(self, capacity, transition_dim, alpha=0.6, beta=0.4, 
                 beta_increment=0.001, epsilon=1e-5):
        """
        初始化优先级经验回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            transition_dim: 转换维度
            alpha: 优先级指数（推荐0.6）
            beta: IS权重初始值（推荐0.4，训练中增长到1.0）
            beta_increment: beta增长速率（推荐0.001）
            epsilon: 优先级最小值（防止为0）
        """
        self.capacity = capacity
        self.transition_dim = transition_dim
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # 使用 float32 节省内存（相比 float64 节省 50%）
        self.buffer = np.zeros((capacity, transition_dim), dtype=np.float32)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        
        # 计数器
        self.counter = 0
        self.max_priority = 1.0  # 新经验的初始优先级
    
    def store(self, state, action, reward, next_state):
        """
        存储一个转换
        
        新存储的经验使用当前最大优先级，确保至少被采样一次
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
        """
        # 构建转换
        transition = np.hstack((state, action, reward, next_state))
        
        # 存储位置（循环覆盖旧数据）
        index = self.counter % self.capacity
        self.buffer[index, :] = transition
        
        # 设置优先级（新经验使用最大优先级，确保被采样）
        self.priorities[index] = self.max_priority
        
        # 更新计数器
        self.counter += 1
    
    def sample(self, batch_size):
        """
        基于优先级采样一批转换
        
        采样概率正比于优先级的alpha次方，高TD-error的经验更容易被采样
        
        Args:
            batch_size: 批次大小
            
        Returns:
            batch: 采样的转换 [batch_size, transition_dim]
            weights: 重要性采样权重 [batch_size, 1]（用于修正偏差）
            indices: 采样的索引（用于更新优先级）
        """
        # 检查是否有足够的样本
        valid_size = min(self.counter, self.capacity)
        assert valid_size >= batch_size, (
            f'记忆库样本不足: {valid_size} < {batch_size}'
        )
        
        # 获取有效的优先级
        valid_priorities = self.priorities[:valid_size]
        
        # 计算采样概率（优先级的 alpha 次方）
        sampling_probs = valid_priorities ** self.alpha
        sampling_probs /= sampling_probs.sum()
        
        # 基于概率采样（不重复）
        indices = np.random.choice(
            valid_size, 
            size=batch_size, 
            replace=False,
            p=sampling_probs
        )
        
        # 获取采样的转换
        batch = self.buffer[indices, :]
        
        # 计算重要性采样权重（修正采样偏差）
        # w_i = (N * P(i))^(-beta) / max_w
        weights = (valid_size * sampling_probs[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化到 [0, 1]
        weights = weights.reshape(-1, 1).astype(np.float32)
        
        # 逐渐增大 beta（从初始值增长到 1.0）
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return batch, weights, indices
    
    def update_priorities(self, indices, priorities):
        """
        更新采样经验的优先级
        
        在每次训练后调用，基于最新的TD-error更新优先级
        
        Args:
            indices: 采样的索引
            priorities: 新的优先级（通常是 |TD-error|）
        """
        # 转换为 numpy 数组（如果是 tensor）
        if hasattr(priorities, 'cpu'):
            priorities = priorities.cpu().detach().numpy()
        
        # 更新优先级（添加 epsilon 防止为0）
        priorities = np.abs(priorities) + self.epsilon
        self.priorities[indices] = priorities.flatten()
        
        # 更新最大优先级（用于新存储的经验）
        self.max_priority = max(self.max_priority, priorities.max())
    
    def is_ready(self, batch_size):
        """
        检查缓冲区是否有足够数据进行采样
        
        Args:
            batch_size: 需要的批次大小
            
        Returns:
            bool: 是否可以采样
        """
        valid_size = min(self.counter, self.capacity)
        return valid_size >= batch_size
    
    def __len__(self):
        """返回缓冲区中有效样本的数量"""
        return min(self.counter, self.capacity)
    
    def get_stats(self):
        """
        获取缓冲区统计信息（用于调试和监控）
        
        Returns:
            dict: 统计信息
        """
        valid_size = min(self.counter, self.capacity)
        valid_priorities = self.priorities[:valid_size]
        
        return {
            'size': valid_size,
            'capacity': self.capacity,
            'usage': valid_size / self.capacity if self.capacity > 0 else 0,
            'max_priority': self.max_priority,
            'mean_priority': valid_priorities.mean() if valid_size > 0 else 0,
            'min_priority': valid_priorities.min() if valid_size > 0 else 0,
            'beta': self.beta
        }
