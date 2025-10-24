"""
配置加载器
用于加载和验证YAML配置文件
"""
import yaml
import os
from typing import Dict, Any


class ConfigLoader:
    """配置加载器，负责读取和验证YAML配置文件"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径，默认为 configs/masac/default.yaml
        """
        if config_path is None:
            # 默认配置文件路径
            project_root = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(project_root, 'configs', 'masac', 'default.yaml')
        
        self.config_path = config_path
        
        # 检查文件是否存在
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
    
    def load(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典
            
        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置验证失败
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"YAML格式错误: {e}")
        
        # 验证配置
        self._validate(config)
        
        # 添加默认值
        config = self._add_defaults(config)
        
        print(f"✓ 成功加载配置文件: {self.config_path}")
        print(f"  - 领导者数量: {config['environment']['n_leaders']}")
        print(f"  - 跟随者数量: {config['environment']['n_followers']}")
        print(f"  - 训练轮数: {config['training']['max_episodes']}")
        
        return config
    
    def _validate(self, config: Dict[str, Any]):
        """
        验证配置的完整性和合法性
        
        Args:
            config: 配置字典
            
        Raises:
            ValueError: 配置验证失败
        """
        # 检查必需的顶层键
        required_keys = ['environment', 'training', 'algorithm']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"配置文件缺少必需的顶层键: {key}")
        
        # 验证环境配置
        env_config = config['environment']
        if 'n_leaders' not in env_config or 'n_followers' not in env_config:
            raise ValueError("环境配置必须包含 'n_leaders' 和 'n_followers'")
        
        n_leaders = env_config['n_leaders']
        n_followers = env_config['n_followers']
        
        # 验证数量范围
        if not isinstance(n_leaders, int) or n_leaders < 1:
            raise ValueError(f"n_leaders 必须是大于0的整数，当前值: {n_leaders}")
        
        if not isinstance(n_followers, int) or n_followers < 1:
            raise ValueError(f"n_followers 必须是大于0的整数，当前值: {n_followers}")
        
        if n_leaders > 10:
            print(f"⚠️  警告: n_leaders={n_leaders} 较大，可能需要更多计算资源")
        
        if n_followers > 10:
            print(f"⚠️  警告: n_followers={n_followers} 较大，可能需要更多计算资源")
        
        # 验证训练配置
        train_config = config['training']
        if train_config.get('max_episodes', 0) <= 0:
            raise ValueError("max_episodes 必须大于0")
        
        if train_config.get('max_steps', 0) <= 0:
            raise ValueError("max_steps 必须大于0")
        
        # 验证算法配置
        algo_config = config['algorithm']
        if not (0 < algo_config.get('gamma', 0.9) <= 1):
            raise ValueError("gamma 必须在 (0, 1] 范围内")
        
        if algo_config.get('batch_size', 0) <= 0:
            raise ValueError("batch_size 必须大于0")
        
        print("✓ 配置验证通过")
    
    def _add_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        为配置添加默认值
        
        Args:
            config: 原始配置字典
            
        Returns:
            添加了默认值的配置字典
        """
        # 环境默认值
        config['environment'].setdefault('render', False)
        config['environment'].setdefault('obstacle_num', 1)
        config['environment'].setdefault('goal_num', 1)
        
        # 训练默认值
        config['training'].setdefault('max_episodes', 500)
        config['training'].setdefault('max_steps', 1000)
        config['training'].setdefault('train_num', 1)
        config['training'].setdefault('test_episodes', 100)
        config['training'].setdefault('switch', 0)
        
        # 算法默认值
        config['algorithm'].setdefault('gamma', 0.9)
        config['algorithm'].setdefault('batch_size', 128)
        config['algorithm'].setdefault('memory_capacity', 20000)
        config['algorithm'].setdefault('tau', 0.01)
        config['algorithm'].setdefault('target_entropy', -0.1)
        
        # 学习率默认值
        if 'learning_rates' not in config['algorithm']:
            config['algorithm']['learning_rates'] = {}
        config['algorithm']['learning_rates'].setdefault('policy_lr', 0.001)
        config['algorithm']['learning_rates'].setdefault('value_lr', 0.003)
        config['algorithm']['learning_rates'].setdefault('q_lr', 0.0003)
        
        # 状态空间默认值
        if 'space' not in config:
            config['space'] = {}
        config['space'].setdefault('state_dim', 7)
        
        # 输出默认值
        if 'output' not in config:
            config['output'] = {}
        config['output'].setdefault('save_interval', 20)
        config['output'].setdefault('save_threshold', 200)
        config['output'].setdefault('output_dir', 'output')
        
        # 噪声默认值
        if 'noise' not in config:
            config['noise'] = {}
        config['noise'].setdefault('ou_noise', True)
        config['noise'].setdefault('noise_episodes', 20)
        
        return config
    
    def to_training_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        将加载的配置转换为训练器所需的格式
        
        Args:
            config: 加载的配置字典
            
        Returns:
            训练器配置字典
        """
        # 获取项目根目录
        project_root = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(project_root, config['output']['output_dir'])
        
        training_config = {
            'n_leaders': config['environment']['n_leaders'],
            'n_followers': config['environment']['n_followers'],
            'state_dim': config['space']['state_dim'],
            'action_dim': 2,  # 由环境决定
            'max_action': 1.0,  # 由环境决定
            'min_action': -1.0,  # 由环境决定
            'gamma': config['algorithm']['gamma'],
            'policy_lr': config['algorithm']['learning_rates']['policy_lr'],
            'value_lr': config['algorithm']['learning_rates']['value_lr'],
            'q_lr': config['algorithm']['learning_rates']['q_lr'],
            'tau': config['algorithm']['tau'],
            'batch_size': config['algorithm']['batch_size'],
            'memory_capacity': config['algorithm']['memory_capacity'],
            'max_episodes': config['training']['max_episodes'],
            'max_steps': config['training']['max_steps'],
            'test_episodes': config['training']['test_episodes'],
            'output_dir': output_dir,
            'target_entropy': config['algorithm']['target_entropy'],
        }
        
        return training_config


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    便捷函数：加载配置文件
    
    Args:
        config_path: 配置文件路径，默认为 configs/masac/default.yaml
        
    Returns:
        配置字典
    """
    loader = ConfigLoader(config_path)
    return loader.load()


if __name__ == '__main__':
    # 测试配置加载
    print("=" * 60)
    print("测试配置加载器")
    print("=" * 60)
    
    # 测试默认配置
    print("\n1. 测试默认配置:")
    try:
        config = load_config()
        print("默认配置加载成功!\n")
    except Exception as e:
        print(f"错误: {e}\n")
    
    # 测试3-follower配置
    print("2. 测试3-follower配置:")
    try:
        config = load_config('configs/example_3followers.yaml')
        print("3-follower配置加载成功!\n")
    except Exception as e:
        print(f"错误: {e}\n")
    
    # 测试5-follower配置
    print("3. 测试5-follower配置:")
    try:
        config = load_config('configs/example_5followers.yaml')
        print("5-follower配置加载成功!\n")
    except Exception as e:
        print(f"错误: {e}\n")
    
    print("=" * 60)

