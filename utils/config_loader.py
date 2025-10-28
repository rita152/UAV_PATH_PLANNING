"""
配置文件加载工具
支持从 YAML 文件加载训练和测试配置
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，默认为 configs/masac/default.yaml
        
    Returns:
        配置字典
        
    Example:
        >>> config = load_config()
        >>> print(config['training']['ep_max'])
        500
    """
    # 获取项目根目录
    if config_path is None:
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / 'configs' / 'masac' / 'default.yaml'
    else:
        config_path = Path(config_path)
    
    # 检查文件是否存在
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    # 加载 YAML 文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def set_env_vars(config: Dict[str, Any]) -> None:
    """
    设置环境变量
    
    Args:
        config: 配置字典
    """
    if 'env_vars' in config:
        for key, value in config['env_vars'].items():
            os.environ[key] = str(value)


def get_train_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取训练参数
    
    Args:
        config: 配置字典
        
    Returns:
        训练参数字典
    """
    params = {}
    
    # 环境参数
    env_config = config.get('environment', {})
    params['n_leader'] = env_config.get('n_leader', 1)
    params['n_follower'] = env_config.get('n_follower', 1)
    params['render'] = env_config.get('render', False)
    params['state_dim'] = env_config.get('state_dim', 7)
    
    # 训练参数
    train_config = config.get('training', {})
    params['ep_max'] = train_config.get('ep_max', 500)
    params['ep_len'] = train_config.get('ep_len', 1000)
    params['train_num'] = train_config.get('train_num', 1)
    params['gamma'] = train_config.get('gamma', 0.9)
    params['batch_size'] = train_config.get('batch_size', 128)
    params['memory_capacity'] = train_config.get('memory_capacity', 20000)
    params['data_save_name'] = train_config.get('data_save_name', 'MASAC_new1.pkl')
    
    # 网络参数
    net_config = config.get('network', {})
    params['hidden_dim'] = net_config.get('hidden_dim', 256)
    params['q_lr'] = net_config.get('q_lr', 3e-4)
    params['value_lr'] = net_config.get('value_lr', 3e-3)
    params['policy_lr'] = net_config.get('policy_lr', 1e-3)
    params['tau'] = net_config.get('tau', 1e-2)
    
    return params


def get_test_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取测试参数
    
    Args:
        config: 配置字典
        
    Returns:
        测试参数字典
    """
    params = {}
    
    # 环境参数
    env_config = config.get('environment', {})
    params['n_leader'] = env_config.get('n_leader', 1)
    params['n_follower'] = env_config.get('n_follower', 1)
    params['state_dim'] = env_config.get('state_dim', 7)
    
    # 测试参数
    test_config = config.get('testing', {})
    params['test_episode'] = test_config.get('test_episode', 100)
    params['ep_len'] = test_config.get('ep_len', 1000)
    params['render'] = test_config.get('render', False)
    params['leader_model_path'] = test_config.get('leader_model_path', None)
    params['follower_model_path'] = test_config.get('follower_model_path', None)
    
    # 网络参数
    net_config = config.get('network', {})
    params['hidden_dim'] = net_config.get('hidden_dim', 256)
    params['policy_lr'] = net_config.get('policy_lr', 1e-3)
    
    return params


def print_config(config: Dict[str, Any]) -> None:
    """
    打印配置信息
    
    Args:
        config: 配置字典
    """
    print("="*60)
    print("配置信息")
    print("="*60)
    
    # 环境配置
    env_config = config.get('environment', {})
    print(f"\n【环境配置】")
    print(f"  Leader 数量:      {env_config.get('n_leader', 1)}")
    print(f"  Follower 数量:    {env_config.get('n_follower', 1)}")
    print(f"  渲染:             {env_config.get('render', False)}")
    
    # 训练配置
    if 'training' in config:
        train_config = config['training']
        print(f"\n【训练配置】")
        print(f"  最大轮数:         {train_config.get('ep_max', 500)}")
        print(f"  每轮步数:         {train_config.get('ep_len', 1000)}")
        print(f"  训练次数:         {train_config.get('train_num', 1)}")
        print(f"  折扣因子:         {train_config.get('gamma', 0.9)}")
        print(f"  批次大小:         {train_config.get('batch_size', 128)}")
        print(f"  经验池容量:       {train_config.get('memory_capacity', 20000)}")
    
    # 测试配置
    if 'testing' in config:
        test_config = config['testing']
        print(f"\n【测试配置】")
        print(f"  测试轮数:         {test_config.get('test_episode', 100)}")
        print(f"  每轮步数:         {test_config.get('ep_len', 1000)}")
    
    # 网络配置
    net_config = config.get('network', {})
    print(f"\n【网络配置】")
    print(f"  隐藏层维度:       {net_config.get('hidden_dim', 256)}")
    print(f"  Q网络学习率:      {net_config.get('q_lr', 3e-4)}")
    print(f"  Value学习率:      {net_config.get('value_lr', 3e-3)}")
    print(f"  Policy学习率:     {net_config.get('policy_lr', 1e-3)}")
    print(f"  软更新系数:       {net_config.get('tau', 1e-2)}")
    
    print("="*60)


if __name__ == '__main__':
    # 测试代码
    config = load_config()
    print_config(config)

