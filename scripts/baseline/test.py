"""
SAC 测试脚本（Baseline版本）
使用 YAML 配置文件管理参数，支持日志保存

使用方法：
    conda activate UAV_PATH_PLANNING
    python scripts/baseline/test.py [--config CONFIG_PATH] [其他可选参数]

参数：
    --config: 配置文件路径（可选），默认使用 configs/masac/default.yaml
    --log_dir: 日志文件存放目录（默认: runs/test_logs）
    --log_name: 日志文件名（默认: test_TIMESTAMP.log）
    其他参数可以覆盖配置文件中的对应参数，例如：
    --leader_model_path runs/exp_xxx/leader.pth
    --follower_model_path runs/exp_xxx/follower.pth
    --test_episode 100 --n_follower 4
"""

import sys
import os
import re
from pathlib import Path
import argparse
from datetime import datetime

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from rl_env.path_env import RlGame
from algorithm.masac import Tester
from utils import load_config, set_env_vars, get_test_params, print_config, get_project_root


class Logger:
    """
    同时输出到终端和文件的日志类
    实时写入，无缓冲
    终端保留颜色，文件去除ANSI颜色代码
    """
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', buffering=1)  # 覆盖写入（测试日志每次重新开始）
        # 编译正则表达式，用于去除ANSI颜色代码
        self.ansi_escape = re.compile(r'\x1b\[[0-9;]*m')

    def write(self, message):
        # 终端输出保留颜色
        self.terminal.write(message)
        # 文件输出去除颜色代码
        clean_message = self.ansi_escape.sub('', message)
        self.log.write(clean_message)
        self.log.flush()  # 强制刷新到磁盘

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# ============================================
# 日志系统
# ============================================

def setup_logging(log_dir=None, log_name=None):
    """
    设置日志系统，将输出重定向到文件和终端

    Args:
        log_dir: 日志目录路径
        log_name: 日志文件名

    Returns:
        logger: Logger实例
        log_file: 日志文件路径
    """
    # 设置默认日志目录
    if log_dir is None:
        log_dir = os.path.join(get_project_root(), 'runs', 'test_logs')

    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)

    # 设置默认日志文件名
    if log_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_name = f'test_{timestamp}.log'

    # 完整的日志文件路径
    log_file = os.path.join(log_dir, log_name)

    # 创建日志器并重定向输出
    logger = Logger(log_file)
    sys.stdout = logger

    # 打印日志信息
    print(f"📝 日志文件: {log_file}")
    print(f"💡 测试输出将实时保存到日志文件")

    return logger, log_file


def cleanup_logging(logger):
    """
    清理日志系统，恢复标准输出

    Args:
        logger: Logger实例

    Returns:
        log_file_path: 日志文件路径
    """
    log_file_path = None
    if logger is not None:
        log_file_path = logger.log.name
        logger.close()
        # 恢复标准输出
        sys.stdout = logger.terminal
    return log_file_path


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

  # 指定日志目录
  python scripts/baseline/test.py --log_dir ./my_test_logs --log_name my_test.log

  # 同时指定配置文件、覆盖参数和日志设置
  python scripts/baseline/test.py --config my_config.yaml --n_follower 2 --log_dir ./results
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

    # 日志参数
    parser.add_argument('--log_dir', type=str, help='日志文件存放目录（默认: runs/test_logs）')
    parser.add_argument('--log_name', type=str, help='日志文件名（默认: test_TIMESTAMP.log）')

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

    # 设置日志系统
    logger, log_file_path = setup_logging(args.log_dir, args.log_name)

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

    # 打印测试表头（与训练格式完全一致）
    print("\n" + "="*80)
    header_parts = ["Episode"]
    # Leader列（只有1个leader，直接使用"Leader"）
    header_parts.append("Leader")
    # Follower列（根据数量添加）
    for j in range(params['n_follower']):
        if params['n_follower'] == 1:
            header_parts.append("Follower")
        else:
            header_parts.append(f"Follower{j}")
    header_parts.append("Steps")
    header_parts.append("Status")
    print(" | ".join([f"{part:^12}" for part in header_parts]))
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

    # 清理日志系统并显示保存消息
    saved_log_path = cleanup_logging(logger)
    print(f"✅ 测试日志已保存: {saved_log_path}")


if __name__ == '__main__':
    main()

