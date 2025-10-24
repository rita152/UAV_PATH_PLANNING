from rl_env.path_env import RlGame
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle as pkl
import sys
import argparse

# 导入MASAC模块
from algorithm.masac import MASACTrainer, MASACTester
# 导入配置加载器和种子管理器
from utils.config_loader import ConfigLoader
from utils.seed_utils import setup_seeds, SeedManager

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='UAV路径规划训练程序')
    parser.add_argument('--config', type=str, default='configs/masac/default.yaml',
                        help='配置文件路径 (默认: configs/masac/default.yaml)')
    parser.add_argument('--n_followers', type=int, default=None,
                        help='跟随者数量（覆盖配置文件）')
    parser.add_argument('--render', action='store_true',
                        help='是否开启可视化渲染')
    parser.add_argument('--test', action='store_true',
                        help='测试模式（默认为训练模式）')
    return parser.parse_args()

# 加载配置
args = parse_args()
config_loader = ConfigLoader(args.config)
yaml_config = config_loader.load()

# 从配置中获取参数
N_Leader = yaml_config['environment']['n_leaders']
N_Follower = yaml_config['environment']['n_followers']
RENDER = yaml_config['environment']['render']
TRAIN_NUM = yaml_config['training']['train_num']
TEST_EPIOSDE = yaml_config['training']['test_episodes']
EP_MAX = yaml_config['training']['max_episodes']
EP_LEN = yaml_config['training']['max_steps']
GAMMA = yaml_config['algorithm']['gamma']
q_lr = yaml_config['algorithm']['learning_rates']['q_lr']
value_lr = yaml_config['algorithm']['learning_rates']['value_lr']
policy_lr = yaml_config['algorithm']['learning_rates']['policy_lr']
BATCH = yaml_config['algorithm']['batch_size']
tau = yaml_config['algorithm']['tau']
MemoryCapacity = yaml_config['algorithm']['memory_capacity']
Switch = yaml_config['training']['switch']

# 命令行参数覆盖
if args.n_followers is not None:
    N_Follower = args.n_followers
    print(f"⚙️  命令行参数覆盖: n_followers={N_Follower}")

if args.render:
    RENDER = True
    print(f"⚙️  命令行参数覆盖: render=True")

if args.test:
    Switch = 1
    print(f"⚙️  命令行参数覆盖: 测试模式")

# 设置随机种子（在创建环境之前）
print(f"\n{'='*60}")
print(f"🎲 随机种子设置")
print(f"{'='*60}")
setup_seeds(yaml_config, episode=0)

# 创建输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, yaml_config['output']['output_dir'])
os.makedirs(OUTPUT_DIR, exist_ok=True)

shoplistfile = os.path.join(OUTPUT_DIR, 'MASAC_new1')
shoplistfile_test = os.path.join(OUTPUT_DIR, 'MASAC_d_test2')
shoplistfile_test1 = os.path.join(OUTPUT_DIR, 'MASAC_compare')

# 创建环境
print(f"\n{'='*60}")
print(f"🚁 初始化UAV路径规划环境")
print(f"{'='*60}")
print(f"  领导者数量: {N_Leader}")
print(f"  跟随者数量: {N_Follower}")
print(f"  总智能体数: {N_Leader + N_Follower}")
print(f"  可视化渲染: {RENDER}")
print(f"  运行模式: {'测试' if Switch == 1 else '训练'}")
print(f"{'='*60}\n")

env = RlGame(n=N_Leader, m=N_Follower, render=RENDER).unwrapped
state_number = 7
action_number = env.action_space.shape[0]
max_action = env.action_space.high[0]
min_action = env.action_space.low[0]

def get_config():
    """获取训练和测试配置"""
    config = {
        'n_leaders': N_Leader,
        'n_followers': N_Follower,
        'state_dim': state_number,
        'action_dim': action_number,
        'max_action': max_action,
        'min_action': min_action,
        'gamma': GAMMA,
        'policy_lr': policy_lr,
        'value_lr': value_lr,
        'q_lr': q_lr,
        'tau': tau,
        'batch_size': BATCH,
        'memory_capacity': MemoryCapacity,
        'max_episodes': EP_MAX,
        'max_steps': EP_LEN,
        'test_episodes': TEST_EPIOSDE,
        'output_dir': OUTPUT_DIR,
        'seed_config': yaml_config.get('seed', {}),  # 添加种子配置
    }
    return config

def main():
    run(env)
def run(env):
    config = get_config()
    
    if Switch == 0:
        # 训练模式
        print('🎓 使用MASACTrainer训练中...')
        print(f"配置信息:")
        print(f"  - 最大轮数: {EP_MAX}")
        print(f"  - 每轮步数: {EP_LEN}")
        print(f"  - 批次大小: {BATCH}")
        print(f"  - 学习率: policy={policy_lr}, value={value_lr}, q={q_lr}")
        print(f"  - 智能体总数: {N_Leader + N_Follower}\n")
        
        # 创建训练器
        trainer = MASACTrainer(env, config)
        
        # 开始训练
        all_rewards = trainer.train()
        
        # 保存训练数据（简化版，仅保存总奖励）
        all_ep_r_mean = np.array(all_rewards)
        d = {"all_ep_r_mean": all_ep_r_mean}
        with open(shoplistfile, 'wb') as f:
            pkl.dump(d, f, pkl.HIGHEST_PROTOCOL)
        
        # 绘制训练曲线
        plt.figure(1, figsize=(8, 4), dpi=150)
        plt.margins(x=0)
        plt.plot(np.arange(len(all_rewards)), all_rewards, label='MASAC', color='#e75840')
        plt.xlabel('Episode')
        plt.ylabel('Total reward')
        plt.legend()
        plt.show()
        
        print(f"\n✅ 训练完成! 共{len(all_rewards)}轮")
        print(f"平均奖励: {np.mean(all_rewards):.2f}")
        print(f"模型已保存到: {OUTPUT_DIR}")
        
        if not RENDER:
            env.close()
    else:
        # 测试模式
        print('🧪 使用MASACTester测试中...')
        # 创建测试器
        tester = MASACTester(env, config)
        
        # 加载模型
        tester.load_models(OUTPUT_DIR)
        
        # 开始测试
        results = tester.test(render=RENDER)
        
        if not RENDER:
            env.close()

if __name__ == '__main__':
    main()