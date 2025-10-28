"""
SAC 测试脚本（Baseline版本）
使用 YAML 配置文件管理参数

使用方法：
    conda activate UAV_PATH_PLANNING
    python scripts/baseline/test.py [--config CONFIG_PATH]

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
from algorithm.masac import Tester
from utils import load_config, set_env_vars, get_test_params, print_config

# ============================================
# 主程序
# ============================================

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MASAC 测试脚本')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（默认: configs/masac/default.yaml）')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置环境变量
    set_env_vars(config)
    
    # 获取测试参数
    params = get_test_params(config)
    
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
    
    # 创建测试器
    tester = Tester(
        env=env,
        n_leader=params['n_leader'],
        n_follower=params['n_follower'],
        state_dim=params['state_dim'],
        action_dim=action_number,
        max_action=max_action,
        min_action=min_action,
        hidden_dim=params['hidden_dim'],
        policy_lr=params['policy_lr'],
        device=params['device'],
        seed=params['seed'],
        leader_model_path=params['leader_model_path'],
        follower_model_path=params['follower_model_path']
    )
    
    # 执行测试
    print("\n" + "="*60)
    print("开始测试 SAC 算法")
    print("="*60)
    
    # 打印配置信息
    print_config(config)
    
    results = tester.test(
        ep_len=params['ep_len'],
        test_episode=params['test_episode'],
        render=params['render']
    )
    
    # 显示详细结果
    print("\n" + "="*60)
    print("测试结果总结")
    print("="*60)
    print(f"任务完成率:        {results['win_rate']:.2%}")
    print(f"平均编队保持率:    {results['average_FKR']:.2%}")
    print(f"平均飞行时间:      {results['average_timestep']:.2f} 步")
    print(f"平均飞行路程:      {results['average_integral_V']:.2f}")
    print(f"平均能量损耗:      {results['average_integral_U']:.2f}")
    print("="*60)


if __name__ == '__main__':
    main()

