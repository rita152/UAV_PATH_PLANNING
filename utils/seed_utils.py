"""
随机种子管理工具
用于设置和管理所有随机数生成器的种子，确保实验可复现
"""
import torch
import numpy as np
import random


def set_seed(seed: int):
    """
    设置所有随机数生成器的种子
    
    Args:
        seed: 种子值
    """
    # Python内置random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况
        
    # 设置PyTorch为确定性模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_episode_seed(base_seed: int, episode: int) -> int:
    """
    根据基础种子和episode编号生成该轮的种子
    
    Args:
        base_seed: 基础种子值
        episode: 当前episode编号
        
    Returns:
        该轮的种子值
    """
    return base_seed + episode


def setup_seeds(config: dict, episode: int = 0):
    """
    根据配置设置种子
    
    Args:
        config: 配置字典
        episode: 当前episode编号（默认0表示初始设置）
    """
    seed_config = config.get('seed', {})
    
    # 检查是否启用种子
    if not seed_config.get('enabled', False):
        print("⚠️  随机种子未启用，实验结果将不可复现")
        return None
    
    base_seed = seed_config.get('base_seed', 42)
    use_episode_seed = seed_config.get('use_episode_seed', True)
    
    if use_episode_seed and episode > 0:
        # 为每轮训练使用不同的种子
        seed = get_episode_seed(base_seed, episode)
    else:
        # 使用基础种子
        seed = base_seed
    
    set_seed(seed)
    
    # 简化输出 - 只在初始化时打印一次
    if episode == 0:
        if use_episode_seed:
            print(f"  - 随机种子: {base_seed} (每轮+1)")
        else:
            print(f"  - 随机种子: {base_seed} (固定)")
    
    return seed


class SeedManager:
    """种子管理器，用于训练过程中管理种子"""
    
    def __init__(self, config: dict):
        """
        初始化种子管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        seed_config = config.get('seed', {})
        
        self.enabled = seed_config.get('enabled', False)
        self.base_seed = seed_config.get('base_seed', 42)
        self.use_episode_seed = seed_config.get('use_episode_seed', True)
        
        if self.enabled:
            # 设置初始种子
            setup_seeds(config, episode=0)
    
    def set_episode_seed(self, episode: int):
        """
        为指定episode设置种子
        
        Args:
            episode: episode编号
            
        Returns:
            当前使用的种子值
        """
        if not self.enabled:
            return None
        
        if self.use_episode_seed:
            seed = get_episode_seed(self.base_seed, episode)
        else:
            seed = self.base_seed
        
        set_seed(seed)
        return seed
    
    def get_info(self) -> dict:
        """
        获取种子管理器信息
        
        Returns:
            包含种子配置信息的字典
        """
        return {
            'enabled': self.enabled,
            'base_seed': self.base_seed,
            'use_episode_seed': self.use_episode_seed,
        }
    
    def __repr__(self):
        if self.enabled:
            mode = "每轮不同种子" if self.use_episode_seed else "固定种子"
            return f"SeedManager(base_seed={self.base_seed}, mode={mode})"
        else:
            return "SeedManager(disabled)"


if __name__ == '__main__':
    # 测试种子设置
    print("测试随机种子管理")
    print("=" * 60)
    
    # 测试1: 基本种子设置
    print("\n1. 测试基本种子设置")
    set_seed(42)
    print(f"   NumPy随机数: {np.random.rand()}")
    print(f"   Python随机数: {random.random()}")
    print(f"   PyTorch随机数: {torch.rand(1).item()}")
    
    # 测试2: 种子复现性
    print("\n2. 测试种子复现性")
    set_seed(42)
    val1 = np.random.rand()
    set_seed(42)
    val2 = np.random.rand()
    print(f"   第一次: {val1}")
    print(f"   第二次: {val2}")
    print(f"   是否相同: {val1 == val2}")
    
    # 测试3: Episode种子生成
    print("\n3. 测试Episode种子生成")
    base_seed = 42
    for ep in range(5):
        seed = get_episode_seed(base_seed, ep)
        print(f"   Episode {ep}: seed={seed}")
    
    # 测试4: 种子管理器
    print("\n4. 测试种子管理器")
    config = {
        'seed': {
            'enabled': True,
            'base_seed': 100,
            'use_episode_seed': True
        }
    }
    manager = SeedManager(config)
    print(f"   管理器信息: {manager}")
    print(f"   详细信息: {manager.get_info()}")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成")

