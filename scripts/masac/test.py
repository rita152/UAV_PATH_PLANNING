"""
UAV路径规划 - MASAC测试脚本
"""
import sys
import os

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from rl_env.path_env import RlGame
import torch
import numpy as np
import argparse

# 导入MASAC模块
from algorithm.masac import MASACTester
# 导入配置加载器、种子管理器和设备管理器
from utils.config_loader import ConfigLoader
from utils.seed_utils import setup_seeds
from utils.device_utils import setup_device

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='UAV路径规划测试程序')
    parser.add_argument('--config', type=str, default='configs/masac/default.yaml',
                        help='配置文件路径 (默认: configs/masac/default.yaml)')
    parser.add_argument('--n_followers', type=int, default=None,
                        help='跟随者数量（覆盖配置文件）')
    parser.add_argument('--render', action='store_true',
                        help='是否开启可视化渲染')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='模型目录路径（默认使用配置文件中的output_dir）')
    parser.add_argument('--test_episodes', type=int, default=None,
                        help='测试轮数（覆盖配置文件）')
    return parser.parse_args()


def load_config(args):
    """
    加载并处理配置
    
    Args:
        args: 命令行参数
        
    Returns:
        yaml_config: YAML配置字典
        env_params: 环境参数字典
        test_params: 测试参数字典
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
    
    test_params = {
        'test_episodes': yaml_config['training']['test_episodes'],
        'model_dir': os.path.join(PROJECT_ROOT, yaml_config['output']['output_dir']),
    }
    
    # 命令行参数覆盖
    if args.n_followers is not None:
        env_params['n_followers'] = args.n_followers
        print(f"⚙️  命令行参数覆盖: n_followers={env_params['n_followers']}")
    
    if args.render:
        env_params['render'] = True
        print(f"⚙️  命令行参数覆盖: render=True")
    
    if args.model_dir is not None:
        test_params['model_dir'] = args.model_dir
        print(f"⚙️  命令行参数覆盖: model_dir={args.model_dir}")
    
    if args.test_episodes is not None:
        test_params['test_episodes'] = args.test_episodes
        print(f"⚙️  命令行参数覆盖: test_episodes={args.test_episodes}")
    
    return yaml_config, env_params, test_params


def create_env(env_params, max_steps=1000):
    """
    创建环境
    
    Args:
        env_params: 环境参数字典
        max_steps: 每个episode的最大步数
        
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
    print(f"  运行模式: 测试")
    print(f"{'='*60}\n")
    
    # ✅ 适配新的环境接口：render参数改为render_mode
    render_mode = 'human' if env_params['render'] else None
    
    # ✅ 修复：传入max_steps参数，从函数参数读取
    env = RlGame(
        n=env_params['n_leaders'],
        m=env_params['n_followers'],
        render_mode=render_mode,
        max_steps=max_steps
    ).unwrapped
    
    env_info = {
        'state_dim': 7,
        'action_dim': env.action_space.shape[0],
        'max_action': env.action_space.high[0],
        'min_action': env.action_space.low[0],
    }
    
    return env, env_info


def get_test_config(yaml_config, env_params, env_info, test_params):
    """
    获取测试配置
    
    Args:
        yaml_config: YAML配置字典
        env_params: 环境参数
        env_info: 环境信息
        test_params: 测试参数
        
    Returns:
        config: 测试配置字典
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
        'test_episodes': test_params['test_episodes'],
        'output_dir': test_params['model_dir'],
        'seed_config': yaml_config.get('seed', {}),
        'device_config': yaml_config.get('device', {}),  # 添加设备配置
    }
    return config


def test(env, config, test_params, env_params):
    """
    执行测试
    
    Args:
        env: 环境实例
        config: 测试配置
        test_params: 测试参数
        env_params: 环境参数
    """
    print('🧪 使用MASACTester测试中...')
    print(f"测试配置:")
    print(f"  - 测试轮数: {config['test_episodes']}")
    print(f"  - 模型目录: {test_params['model_dir']}")
    print(f"  - 智能体总数: {config['n_leaders'] + config['n_followers']}")
    print(f"  - 可视化: {env_params['render']}\n")
    
    # 创建测试器
    tester = MASACTester(env, config)
    
    # 加载模型
    print(f"📂 加载模型从: {test_params['model_dir']}")
    tester.load_models(test_params['model_dir'])
    
    # 开始测试
    print(f"\n🎮 开始测试...")
    results = tester.test(render=env_params['render'])
    
    print(f"\n✅ 测试完成!")
    
    # 关闭环境
    if not env_params['render']:
        env.close()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 加载配置
    yaml_config, env_params, test_params = load_config(args)
    
    # 设置计算设备
    print(f"\n{'='*60}")
    print(f"🖥️  计算设备设置")
    print(f"{'='*60}")
    device = setup_device(yaml_config)
    
    # 设置随机种子（测试时也可以设置种子以保证可复现）
    print(f"\n{'='*60}")
    print(f"🎲 随机种子设置")
    print(f"{'='*60}")
    setup_seeds(yaml_config, episode=0)
    
    # 创建环境（传入max_steps）
    env, env_info = create_env(env_params, max_steps=yaml_config['training']['max_steps'])
    
    # 获取测试配置
    config = get_test_config(yaml_config, env_params, env_info, test_params)
    
    # 执行测试
    test(env, config, test_params, env_params)


if __name__ == '__main__':
    main()

