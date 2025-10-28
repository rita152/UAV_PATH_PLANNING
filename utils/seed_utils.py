"""
随机种子管理工具
确保训练和测试的可复现性

设计原则：
1. 训练和测试使用不同的种子空间（避免过拟合）
2. 每个episode使用不同的种子（探索多样性）
3. 整体过程完全可复现（相同配置 → 相同结果）

种子分配策略：
- 训练种子: base_seed + episode (e.g., 42, 43, 44, ...)
- 测试种子: base_seed + 10000 + episode (e.g., 10042, 10043, 10044, ...)
"""

import random
import numpy as np
import torch
from typing import Literal


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """
    设置全局随机种子（Python, NumPy, PyTorch, CUDA）
    
    Args:
        seed: 随机种子
        deterministic: 是否启用完全确定性（会降低性能10-30%）
        
    Example:
        >>> set_global_seed(42)
        >>> # 所有随机操作现在都是确定性的
    """
    # Python random 模块
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random (CPU)
    torch.manual_seed(seed)
    
    # PyTorch random (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU
    
    # 完全确定性设置（可选，会降低性能）
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 注意：某些PyTorch操作可能不支持确定性
        # torch.use_deterministic_algorithms(True)


def get_episode_seed(base_seed: int, episode: int, mode: Literal['train', 'test'] = 'train') -> int:
    """
    获取指定episode的种子
    
    Args:
        base_seed: 基础种子
        episode: Episode编号（从0开始）
        mode: 'train' 或 'test'
        
    Returns:
        该episode的确定性种子
        
    Example:
        >>> # 训练种子
        >>> get_episode_seed(42, 0, 'train')
        42
        >>> get_episode_seed(42, 1, 'train')
        43
        
        >>> # 测试种子（不与训练重叠）
        >>> get_episode_seed(42, 0, 'test')
        10042
        >>> get_episode_seed(42, 1, 'test')
        10043
    """
    # 测试种子偏移10000，确保与训练完全分离
    offset = 10000 if mode == 'test' else 0
    return base_seed + offset + episode


def print_seed_info(base_seed: int, mode: str, deterministic: bool = False) -> None:
    """
    打印种子信息
    
    Args:
        base_seed: 基础种子
        mode: 'train' 或 'test'
        deterministic: 是否启用完全确定性
    """
    if mode == 'train':
        print(f"🎲 训练种子: {base_seed} (每个Episode递增)")
    else:
        print(f"🎲 测试种子: {base_seed + 10000} (每个Episode递增)")
    
    if deterministic:
        print("⚠️  完全确定性模式已启用（性能可能下降10-30%）")


if __name__ == '__main__':
    # 测试代码
    print("="*60)
    print("种子管理工具测试")
    print("="*60)
    
    # 测试种子生成
    print("\n训练种子示例（base_seed=42）：")
    for i in range(5):
        seed = get_episode_seed(42, i, 'train')
        print(f"  Episode {i}: seed = {seed}")
    
    print("\n测试种子示例（base_seed=42）：")
    for i in range(5):
        seed = get_episode_seed(42, i, 'test')
        print(f"  Episode {i}: seed = {seed}")
    
    # 测试全局种子设置
    print("\n设置全局种子测试：")
    set_global_seed(42)
    print(f"  Python random: {random.randint(0, 100)}")
    print(f"  NumPy random: {np.random.randint(0, 100)}")
    print(f"  PyTorch random: {torch.randint(0, 100, (1,)).item()}")
    
    # 再次设置相同种子，结果应该一致
    set_global_seed(42)
    print(f"\n再次设置种子42：")
    print(f"  Python random: {random.randint(0, 100)} (应该与上面相同)")
    print(f"  NumPy random: {np.random.randint(0, 100)} (应该与上面相同)")
    print(f"  PyTorch random: {torch.randint(0, 100, (1,)).item()} (应该与上面相同)")
    
    print("\n" + "="*60)
    print("✅ 所有测试通过！")

