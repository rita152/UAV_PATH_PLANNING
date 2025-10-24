"""
设备管理工具
用于管理CPU/GPU设备，支持CUDA加速
"""
import torch


def get_device(config: dict = None) -> torch.device:
    """
    根据配置获取计算设备
    
    Args:
        config: 配置字典，包含device配置
        
    Returns:
        device: PyTorch设备对象
    """
    if config is None:
        config = {}
    
    device_config = config.get('device', {})
    use_cuda = device_config.get('use_cuda', True)
    cuda_device = device_config.get('cuda_device', 0)
    
    # 检查CUDA是否可用
    if use_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{cuda_device}')
        print(f"✓ 使用GPU设备: {torch.cuda.get_device_name(cuda_device)}")
        print(f"  - CUDA版本: {torch.version.cuda}")
        print(f"  - 设备ID: cuda:{cuda_device}")
        print(f"  - 显存容量: {torch.cuda.get_device_properties(cuda_device).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device('cpu')
        if use_cuda and not torch.cuda.is_available():
            print(f"⚠️  配置要求使用CUDA，但CUDA不可用，降级使用CPU")
        else:
            print(f"✓ 使用CPU设备")
    
    return device


def get_device_info() -> dict:
    """
    获取设备信息
    
    Returns:
        info: 包含设备信息的字典
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        info['devices'] = []
        for i in range(torch.cuda.device_count()):
            device_info = {
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'total_memory_gb': torch.cuda.get_device_properties(i).total_memory / 1024**3,
                'compute_capability': torch.cuda.get_device_properties(i).major,
            }
            info['devices'].append(device_info)
    
    return info


def print_device_info():
    """打印设备信息"""
    print("\n" + "="*60)
    print("🖥️  设备信息")
    print("="*60)
    
    info = get_device_info()
    
    print(f"CUDA可用: {'是' if info['cuda_available'] else '否'}")
    
    if info['cuda_available']:
        print(f"CUDA版本: {info['cuda_version']}")
        print(f"GPU数量: {info['cuda_device_count']}")
        print(f"\nGPU详细信息:")
        for device in info['devices']:
            print(f"  GPU {device['id']}: {device['name']}")
            print(f"    显存: {device['total_memory_gb']:.2f} GB")
            print(f"    计算能力: {device['compute_capability']}.x")
    else:
        print("提示: 安装CUDA版本的PyTorch可以加速训练")
        print("安装命令: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    
    print("="*60)


def setup_device(config: dict = None) -> torch.device:
    """
    设置并返回计算设备
    
    Args:
        config: 配置字典
        
    Returns:
        device: PyTorch设备对象
    """
    device = get_device(config)
    
    # 如果使用CUDA，设置一些优化选项
    if device.type == 'cuda':
        # 启用cudnn自动寻找最优算法
        torch.backends.cudnn.benchmark = True
        # 清空缓存
        torch.cuda.empty_cache()
    
    return device


if __name__ == '__main__':
    # 测试设备工具
    print("测试设备管理工具")
    print_device_info()
    
    # 测试设备获取
    print("\n测试设备获取:")
    
    # 测试1: 启用CUDA
    config1 = {
        'device': {
            'use_cuda': True,
            'cuda_device': 0
        }
    }
    device1 = get_device(config1)
    print(f"配置1结果: {device1}")
    
    # 测试2: 禁用CUDA
    config2 = {
        'device': {
            'use_cuda': False,
        }
    }
    device2 = get_device(config2)
    print(f"配置2结果: {device2}")
    
    # 测试3: 无配置（使用默认值）
    device3 = get_device()
    print(f"配置3结果: {device3}")

