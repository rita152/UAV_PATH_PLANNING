"""
SAC 训练脚本（Baseline版本）
功能与 commit 77304be 中的 main_SAC.py 训练模式完全一致

使用方法：
    conda activate UAV_PATH_PLANNING
    python scripts/baseline/train.py
"""

import sys
import os
from pathlib import Path

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from rl_env.path_env import RlGame
from algorithm.masac import Trainer

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

# 训练次数
TRAIN_NUM = 1

# 创建环境
env = RlGame(n=N_LEADER, m=N_FOLLOWER, render=RENDER).unwrapped

# 状态和动作空间参数
state_number = 7
action_number = env.action_space.shape[0]
max_action = env.action_space.high[0]
min_action = env.action_space.low[0]

# 训练参数
EP_MAX = 500        # 最大训练轮数
EP_LEN = 1000       # 每轮最大步数
GAMMA = 0.9         # 折扣因子
BATCH = 128         # 批次大小
MemoryCapacity = 20000  # 经验池容量

# 网络参数
HIDDEN_DIM = 256    # 隐藏层维度
q_lr = 3e-4         # Q网络学习率
value_lr = 3e-3     # Value网络学习率
policy_lr = 1e-3    # Policy学习率
tau = 1e-2          # 软更新系数

# ============================================
# 主程序
# ============================================

def main():
    """主函数"""
    # 检查 N_FOLLOWER 约束
    try:
        assert N_FOLLOWER == 1
    except AssertionError:
        print('程序终止，被逮到~嘿嘿，哥们儿预判到你会犯错，这段程序中变量\'N_FOLLOWER\'的值必须为1，请把它的值改为1。\n' 
              '改为1之后程序一定会报错，这是因为组数越界，更改path_env.py文件中的跟随者无人机初始化个数；删除多余的\n'
              '求距离函数，即变量dis_1_agent_0_to_3等，以及提到变量dis_1_agent_0_to_3等的地方；删除画无人机轨迹的\n'
              '函数；删除step函数的最后一个返回值dis_1_agent_0_to_1；将player.py文件中的变量dt改为1；即可开始训练！\n'
              '如果实在不会改也无妨，我会在不久之后出一个视频来手把手教大伙怎么改，可持续关注此项目github中的README文件。\n')
        return
    
    # 创建训练器（配置参数）
    trainer = Trainer(
        env=env,
        n_leader=N_LEADER,
        n_follower=N_FOLLOWER,
        state_dim=state_number,
        action_dim=action_number,
        max_action=max_action,
        min_action=min_action,
        hidden_dim=HIDDEN_DIM,
        gamma=GAMMA,
        q_lr=q_lr,
        value_lr=value_lr,
        policy_lr=policy_lr,
        tau=tau,
        batch_size=BATCH,
        memory_capacity=MemoryCapacity,
        data_save_name='MASAC_new1.pkl'
    )
    
    # 执行训练（运行时参数）
    print("="*50)
    print("开始训练 SAC 算法")
    print("="*50)
    print(f"Leader数量: {N_LEADER}")
    print(f"Follower数量: {N_FOLLOWER}")
    print(f"最大训练轮数: {EP_MAX}")
    print(f"每轮最大步数: {EP_LEN}")
    print(f"训练次数: {TRAIN_NUM}")
    print("="*50)
    
    data = trainer.train(
        ep_max=EP_MAX,
        ep_len=EP_LEN,
        train_num=TRAIN_NUM,
        render=RENDER
    )
    
    print("\n训练完成！")
    print(f"训练数据已保存到: saved_models/data/MASAC_new1.pkl")
    print(f"模型已保存到: saved_models/Path_SAC_actor_L1.pth")
    print(f"模型已保存到: saved_models/Path_SAC_actor_F1.pth")


if __name__ == '__main__':
    main()

