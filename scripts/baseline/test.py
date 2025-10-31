"""
SAC 测试脚本（Baseline版本）
使用 YAML 配置文件管理参数

使用方法：
    conda activate UAV_PATH_PLANNING
    python scripts/baseline/test.py [--config CONFIG_PATH] [其他可选参数]

参数：
    --config: 配置文件路径（可选），默认使用 configs/masac/default.yaml
    其他参数可以覆盖配置文件中的对应参数，例如：
    --leader_model_path runs/exp_xxx/leader.pth
    --follower_model_path runs/exp_xxx/follower.pth
    --test_episode 100 --n_follower 4
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
    parser = argparse.ArgumentParser(
        description='MASAC 测试脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 使用默认配置
  python scripts/baseline/test.py

  # 指定配置文件
  python scripts/baseline/test.py --config configs/masac/default.yaml

  # 覆盖配置中的参数
  python scripts/baseline/test.py --leader_model_path runs/exp_xxx/leader.pth --test_episode 100

  # 同时指定配置文件和覆盖参数
  python scripts/baseline/test.py --config my_config.yaml --n_follower 2 --render
        """
    )

    # 配置文件路径
    parser.add_argument('--config', type=str, default='configs/masac/default.yaml',
                       help='配置文件路径（默认: configs/masac/default.yaml）')

    # 测试参数覆盖
    parser.add_argument('--test_episode', type=int, help='测试轮数')
    parser.add_argument('--ep_len', type=int, help='每轮最大步数')
    parser.add_argument('--render', action='store_true', help='是否渲染可视化')

    # 模型路径覆盖（重点功能）
    parser.add_argument('--leader_model_path', type=str, help='Leader模型权重文件路径')
    parser.add_argument('--follower_model_path', type=str, help='Follower模型权重文件路径')

    # 环境参数覆盖
    parser.add_argument('--n_leader', type=int, help='Leader数量')
    parser.add_argument('--n_follower', type=int, help='Follower数量')
    parser.add_argument('--state_dim', type=int, help='状态维度')

    # 设备和种子参数覆盖
    parser.add_argument('--device', type=str, help='测试设备 (auto/cpu/cuda/cuda:0)')
    parser.add_argument('--seed', type=int, help='随机种子')

    # 网络参数覆盖
    parser.add_argument('--hidden_dim', type=int, help='隐藏层维度')
    parser.add_argument('--policy_lr', type=float, help='策略网络学习率')

    args = parser.parse_args()

    # 构建参数覆盖字典（只包含用户指定的参数）
    overrides = {}
    for key, value in vars(args).items():
        if key != 'config' and value is not None:
            overrides[key] = value

    # 加载配置
    config = load_config(args.config)

    # 应用命令行参数覆盖
    def apply_overrides(config_dict, overrides_dict, prefix=''):
        """递归应用参数覆盖"""
        for key, value in overrides_dict.items():
            if isinstance(value, dict):
                if key in config_dict:
                    apply_overrides(config_dict[key], value, f"{prefix}{key}.")
                else:
                    config_dict[key] = value
            else:
                # 尝试在不同配置section中查找并设置
                sections = ['environment', 'training', 'testing', 'network', 'output']
                found = False
                for section in sections:
                    if section in config_dict and key in config_dict[section]:
                        config_dict[section][key] = value
                        found = True
                        break
                if not found:
                    # 如果没找到对应的section，尝试直接设置到根级别
                    config_dict[key] = value

    if overrides:
        apply_overrides(config, overrides)

    # 设置环境变量
    set_env_vars(config)

    # 获取测试参数
    params = get_test_params(config)
    
    # 创建环境
    env = RlGame(
        n_leader=params['n_leader'],
        n_follower=params['n_follower'],
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
    print(f"📄 配置文件: {args.config}")
    if overrides:
        print(f"🔧 参数覆盖: {overrides}")
    print("="*60)

    # 打印配置信息
    print_config(config)

    # 打印测试表头（与训练格式保持一致的风格）
    print("\n" + "="*80)
    print("  Episode    |    Score    |   Steps    |    Status")
    print("="*80)

    print('SAC测试中...')

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

