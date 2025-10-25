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
    parser.add_argument('--output_dir', type=str, default=None,
                        help='测试结果输出目录（默认与model_dir相同）')
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
        'output_dir': None,  # 默认为None，后续根据model_dir设置
    }
    
    # 命令行参数覆盖
    if args.n_followers is not None:
        env_params['n_followers'] = args.n_followers
    
    if args.render:
        env_params['render'] = True
    
    if args.model_dir is not None:
        test_params['model_dir'] = args.model_dir
    
    if args.output_dir is not None:
        test_params['output_dir'] = args.output_dir
    elif args.model_dir is not None:
        # 默认output_dir与model_dir相同
        test_params['output_dir'] = args.model_dir
    else:
        test_params['output_dir'] = test_params['model_dir']
    
    if args.test_episodes is not None:
        test_params['test_episodes'] = args.test_episodes
    
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
        'seed_config': yaml_config.get('seed', {}),  # 包含修改后的base_seed=10000
        'device_config': yaml_config.get('device', {}),
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
    # 设置输出目录
    output_dir = test_params['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置测试日志
    from utils.logger_utils import setup_logger
    logger, log_file = setup_logger(output_dir, name='testing')
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🧪 开始测试")
    logger.info(f"{'='*60}")
    logger.info(f"  Leader × {config['n_leaders']} | Follower × {config['n_followers']} | Episodes: {config['test_episodes']}")
    logger.info(f"  模型目录: {test_params['model_dir']}")
    logger.info(f"  测试日志: {log_file}")
    logger.info(f"{'='*60}\n")
    
    # ✅ 先将logger传入config，再创建测试器
    config['logger'] = logger
    
    # 创建测试器（此时logger已在config中）
    tester = MASACTester(env, config)
    
    # 加载模型（静默加载，只输出关键信息）
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    tester.load_models(test_params['model_dir'])
    sys.stdout = old_stdout
    
    logger.info("✓ 模型加载成功")
    
    # 开始测试
    results = tester.test(render=env_params['render'])
    
    # 保存测试结果
    import json
    import datetime
    
    # 准备可序列化的结果（转换numpy数组为list）
    serializable_results = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'success_rate': float(results['success_rate']),
        'total_avg_reward': float(results['total_avg_reward']),
        'leader_avg_reward': float(results['leader_avg_reward']),
        'follower_avg_reward': float(results['follower_avg_reward']),
        'std_reward': float(results['std_reward']),
        'avg_steps': float(results['avg_steps']),
        'avg_formation_keeping': float(results['avg_formation_keeping']),
        'avg_flight_distance': float(results['avg_flight_distance']),
        'avg_energy_consumption': float(results['avg_energy_consumption']),
        'total_episodes': int(results['total_episodes']),
        'win_count': int(results['win_count']),
        'all_rewards': [float(x) for x in results['all_rewards']],
        'all_leader_rewards': [float(x) for x in results['all_leader_rewards']],
        'all_follower_rewards': [float(x) for x in results['all_follower_rewards']],
    }
    
    result_file = os.path.join(output_dir, f'test_results_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✅ 测试完成!")
    logger.info(f"测试结果已保存: {result_file}")
    logger.info(f"测试日志已保存: {log_file}")
    
    # 关闭环境
    if not env_params['render']:
        env.close()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 加载配置
    yaml_config, env_params, test_params = load_config(args)
    
    # 设置计算设备（简化输出）
    device = setup_device(yaml_config)
    
    # 设置随机种子（使用不同于训练的种子范围，避免数据泄漏）
    # 训练种子范围: [42, 541] (500 episodes)
    # 测试种子范围: [10000, ...] (确保不重叠)
    if 'seed' in yaml_config and yaml_config['seed'].get('enabled', False):
        yaml_config['seed']['base_seed'] = 10000  # 修改测试的基础种子
    setup_seeds(yaml_config, episode=0)
    
    # 创建环境（传入max_steps）
    env, env_info = create_env(env_params, max_steps=yaml_config['training']['max_steps'])
    
    # 获取测试配置
    config = get_test_config(yaml_config, env_params, env_info, test_params)
    
    # 执行测试
    test(env, config, test_params, env_params)


if __name__ == '__main__':
    main()

