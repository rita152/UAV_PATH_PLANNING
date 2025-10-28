"""
SAC 测试脚本（Baseline版本）
功能与 commit 77304be 中的 main_SAC.py 测试模式完全一致

使用方法：
    conda activate UAV_PATH_PLANNING
    python scripts/baseline/test.py
"""

import sys
import os
from pathlib import Path

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from rl_env.path_env import RlGame
from algorithm.masac import Tester

# 环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ============================================
# 参数配置（与原始 main_SAC.py 完全一致）
# ============================================

# 智能体数量（对应原始的 N_Agent=1, M_Enemy=1）
N_LEADER = 1
N_FOLLOWER = 1

# 渲染设置
RENDER = False

# 测试参数
TEST_EPIOSDE = 100  # 测试轮数

# 创建环境
env = RlGame(n=N_LEADER, m=N_FOLLOWER, render=RENDER).unwrapped

# 状态和动作空间参数
state_number = 7
action_number = env.action_space.shape[0]
max_action = env.action_space.high[0]
min_action = env.action_space.low[0]

# 测试参数
EP_LEN = 1000       # 每轮最大步数

# 网络参数（必须与训练时一致）
HIDDEN_DIM = 256    # 隐藏层维度
policy_lr = 1e-3    # Policy学习率

# ============================================
# 主程序
# ============================================

def main():
    """主函数"""
    # 创建测试器（配置参数）
    tester = Tester(
        env=env,
        n_leader=N_LEADER,
        n_follower=N_FOLLOWER,
        state_dim=state_number,
        action_dim=action_number,
        max_action=max_action,
        min_action=min_action,
        hidden_dim=HIDDEN_DIM,
        policy_lr=policy_lr,
        leader_model_path=None,  # 使用默认路径
        follower_model_path=None
    )
    
    # 执行测试（运行时参数）
    print("="*50)
    print("开始测试 SAC 算法")
    print("="*50)
    print(f"Leader数量: {N_LEADER}")
    print(f"Follower数量: {N_FOLLOWER}")
    print(f"测试轮数: {TEST_EPIOSDE}")
    print(f"每轮最大步数: {EP_LEN}")
    print("="*50)
    
    results = tester.test(
        ep_len=EP_LEN,
        test_episode=TEST_EPIOSDE,
        render=RENDER
    )
    
    # 显示详细结果
    print("\n" + "="*50)
    print("测试结果总结")
    print("="*50)
    print(f"任务完成率:        {results['win_rate']:.2%}")
    print(f"平均编队保持率:    {results['average_FKR']:.2%}")
    print(f"平均飞行时间:      {results['average_timestep']:.2f} 步")
    print(f"平均飞行路程:      {results['average_integral_V']:.2f}")
    print(f"平均能量损耗:      {results['average_integral_U']:.2f}")
    print("="*50)


if __name__ == '__main__':
    main()

