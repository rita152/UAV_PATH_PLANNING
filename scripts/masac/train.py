"""
UAV路径规划 - MASAC训练脚本
"""
import sys
import os

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from rl_env.path_env import RlGame
import torch
import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl
import argparse

# 导入MASAC模块
from algorithm.masac import MASACTrainer
# 导入配置加载器、种子管理器和设备管理器
from utils.config_loader import ConfigLoader
from utils.seed_utils import setup_seeds
from utils.device_utils import setup_device

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='UAV路径规划训练程序')
    parser.add_argument('--config', type=str, default='configs/masac/default.yaml',
                        help='配置文件路径 (默认: configs/masac/default.yaml)')
    parser.add_argument('--n_followers', type=int, default=None,
                        help='跟随者数量（覆盖配置文件）')
    parser.add_argument('--render', action='store_true',
                        help='是否开启可视化渲染')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（覆盖配置文件）')
    return parser.parse_args()


def load_config(args):
    """
    加载并处理配置
    
    Args:
        args: 命令行参数
        
    Returns:
        yaml_config: YAML配置字典
        env_params: 环境参数字典
    """
    # 加载配置
    config_path = os.path.join(PROJECT_ROOT, args.config)
    config_loader = ConfigLoader(config_path)
    yaml_config = config_loader.load()
    
    # 从配置中获取参数
    env_params = {
        'n_leaders': yaml_config['environment']['n_leaders'],
        'n_followers': yaml_config['environment']['n_followers'],
        'render': yaml_config['environment']['render'],
    }
    
    # 命令行参数覆盖
    if args.n_followers is not None:
        env_params['n_followers'] = args.n_followers
        print(f"⚙️  命令行参数覆盖: n_followers={env_params['n_followers']}")
    
    if args.render:
        env_params['render'] = True
        print(f"⚙️  命令行参数覆盖: render=True")
    
    if args.output_dir is not None:
        yaml_config['output']['output_dir'] = args.output_dir
        print(f"⚙️  命令行参数覆盖: output_dir={args.output_dir}")
    
    return yaml_config, env_params


def setup_output_dir(yaml_config):
    """
    创建输出目录
    
    Args:
        yaml_config: YAML配置字典
        
    Returns:
        output_dir: 输出目录路径
        shoplistfile: 训练数据保存路径
    """
    output_dir = os.path.join(PROJECT_ROOT, yaml_config['output']['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    shoplistfile = os.path.join(output_dir, 'MASAC_new1')
    
    return output_dir, shoplistfile


def create_env(env_params):
    """
    创建环境
    
    Args:
        env_params: 环境参数字典
        
    Returns:
        env: 环境实例
        env_info: 环境信息字典
    """
    print(f"\n{'='*60}")
    print(f"🚁 初始化UAV路径规划环境")
    print(f"{'='*60}")
    print(f"  领导者数量: {env_params['n_leaders']}")
    print(f"  跟随者数量: {env_params['n_followers']}")
    print(f"  总智能体数: {env_params['n_leaders'] + env_params['n_followers']}")
    print(f"  可视化渲染: {env_params['render']}")
    print(f"  运行模式: 训练")
    print(f"{'='*60}\n")
    
    env = RlGame(
        n=env_params['n_leaders'],
        m=env_params['n_followers'],
        render=env_params['render']
    ).unwrapped
    
    env_info = {
        'state_dim': 7,
        'action_dim': env.action_space.shape[0],
        'max_action': env.action_space.high[0],
        'min_action': env.action_space.low[0],
    }
    
    return env, env_info


def get_training_config(yaml_config, env_params, env_info, output_dir):
    """
    获取训练配置
    
    Args:
        yaml_config: YAML配置字典
        env_params: 环境参数
        env_info: 环境信息
        output_dir: 输出目录
        
    Returns:
        config: 训练配置字典
    """
    config = {
        'n_leaders': env_params['n_leaders'],
        'n_followers': env_params['n_followers'],
        'state_dim': env_info['state_dim'],
        'action_dim': env_info['action_dim'],
        'max_action': env_info['max_action'],
        'min_action': env_info['min_action'],
        'gamma': yaml_config['algorithm']['gamma'],
        'policy_lr': yaml_config['algorithm']['learning_rates']['policy_lr'],
        'value_lr': yaml_config['algorithm']['learning_rates']['value_lr'],
        'q_lr': yaml_config['algorithm']['learning_rates']['q_lr'],
        'tau': yaml_config['algorithm']['tau'],
        'batch_size': yaml_config['algorithm']['batch_size'],
        'memory_capacity': yaml_config['algorithm']['memory_capacity'],
        'max_episodes': yaml_config['training']['max_episodes'],
        'max_steps': yaml_config['training']['max_steps'],
        'test_episodes': yaml_config['training']['test_episodes'],
        'output_dir': output_dir,
        'seed_config': yaml_config.get('seed', {}),
        'device_config': yaml_config.get('device', {}),  # 添加设备配置
    }
    return config


def train(env, config, shoplistfile, env_params):
    """
    执行训练
    
    Args:
        env: 环境实例
        config: 训练配置
        shoplistfile: 训练数据保存路径
        env_params: 环境参数
    """
    print('🎓 使用MASACTrainer训练中...')
    print(f"配置信息:")
    print(f"  - 最大轮数: {config['max_episodes']}")
    print(f"  - 每轮步数: {config['max_steps']}")
    print(f"  - 批次大小: {config['batch_size']}")
    print(f"  - 学习率: policy={config['policy_lr']}, value={config['value_lr']}, q={config['q_lr']}")
    print(f"  - 智能体总数: {config['n_leaders'] + config['n_followers']}\n")
    
    # 创建训练器
    trainer = MASACTrainer(env, config)
    
    # 开始训练
    all_rewards = trainer.train()
    
    # 保存训练数据
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
    
    # 保存训练曲线图（避免plt.show()阻塞训练）
    curve_path = os.path.join(config['output_dir'], 'training_curve.png')
    plt.savefig(curve_path)
    plt.close()  # 关闭图表释放内存
    print(f"训练曲线已保存到: {curve_path}")
    
    print(f"\n✅ 训练完成! 共{len(all_rewards)}轮")
    print(f"平均奖励: {np.mean(all_rewards):.2f}")
    print(f"最佳奖励: {np.max(all_rewards):.2f}")
    print(f"模型已保存到: {config['output_dir']}")
    print(f"训练数据已保存到: {shoplistfile}")
    
    # 关闭环境
    if not env_params['render']:
        env.close()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 加载配置
    yaml_config, env_params = load_config(args)
    
    # 设置计算设备
    print(f"\n{'='*60}")
    print(f"🖥️  计算设备设置")
    print(f"{'='*60}")
    device = setup_device(yaml_config)
    
    # 设置随机种子
    print(f"\n{'='*60}")
    print(f"🎲 随机种子设置")
    print(f"{'='*60}")
    setup_seeds(yaml_config, episode=0)
    
    # 设置输出目录
    output_dir, shoplistfile = setup_output_dir(yaml_config)
    
    # 创建环境
    env, env_info = create_env(env_params)
    
    # 获取训练配置
    config = get_training_config(yaml_config, env_params, env_info, output_dir)
    
    # 执行训练
    train(env, config, shoplistfile, env_params)


if __name__ == '__main__':
    main()

