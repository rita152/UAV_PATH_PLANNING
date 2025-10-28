"""
SAC 训练脚本（Baseline版本）
使用 YAML 配置文件管理参数

使用方法：
    conda activate UAV_PATH_PLANNING
    python scripts/baseline/train.py [--config CONFIG_PATH]

参数：
    --config: 配置文件路径（可选），默认使用 configs/masac/default.yaml
"""

import sys
import os
from pathlib import Path
import argparse

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from rl_env.path_env import RlGame
from algorithm.masac import Trainer
from utils import load_config, set_env_vars, get_train_params, print_config

# ============================================
# 主程序
# ============================================

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MASAC 训练脚本')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（默认: configs/masac/default.yaml）')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置环境变量
    set_env_vars(config)
    
    # 获取训练参数
    params = get_train_params(config)
    
    # 创建环境
    env = RlGame(
        n=params['n_leader'],
        m=params['n_follower'],
        render=params['render']
    ).unwrapped
    
    # 从环境获取动作空间参数
    action_number = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]
    
    # 创建训练器
    trainer = Trainer(
        env=env,
        n_leader=params['n_leader'],
        n_follower=params['n_follower'],
        state_dim=params['state_dim'],
        action_dim=action_number,
        max_action=max_action,
        min_action=min_action,
        hidden_dim=params['hidden_dim'],
        gamma=params['gamma'],
        q_lr=params['q_lr'],
        value_lr=params['value_lr'],
        policy_lr=params['policy_lr'],
        tau=params['tau'],
        batch_size=params['batch_size'],
        memory_capacity=params['memory_capacity'],
        device=params['device'],
        seed=params['seed'],
        deterministic=params['deterministic'],
        data_save_name=params['data_save_name']
    )
    
    # 执行训练
    print("\n" + "="*60)
    print("开始训练 SAC 算法")
    print("="*60)
    
    # 打印配置信息
    print_config(config)
    
    data = trainer.train(
        ep_max=params['ep_max'],
        ep_len=params['ep_len'],
        train_num=params['train_num'],
        render=params['render']
    )
    
    # 训练完成
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"训练数据: saved_models/data/{params['data_save_name']}")
    print(f"Leader模型: saved_models/leader.pth")
    if params['n_follower'] > 0:
        print(f"Follower模型: saved_models/follower.pth (所有Follower共享)")
    print("="*60)


if __name__ == '__main__':
    main()

