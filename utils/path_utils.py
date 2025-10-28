"""
路径管理工具模块

提供项目路径的统一管理，确保项目可以在任何设备上运行。
所有路径都基于项目根目录，自动转换为绝对路径。

使用示例：
    from utils import get_abs_path, ensure_dir, get_model_path
    
    # 获取绝对路径
    config_path = get_abs_path('config/settings.json')
    
    # 确保目录存在
    save_dir = ensure_dir('saved_models')
    
    # 获取模型保存路径
    model_path = get_model_path('actor_L1.pth')
"""

import os
from pathlib import Path
from typing import Union


def get_project_root() -> str:
    """
    获取项目根目录的绝对路径
    
    Returns:
        str: 项目根目录的绝对路径
        
    Example:
        >>> root = get_project_root()
        >>> print(root)
        /home/user/UAV_PATH_PLANNING
    """
    # 从当前文件向上查找项目根目录
    # utils/path_utils.py -> utils/ -> 项目根目录
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    return str(project_root)


def get_abs_path(relative_path: Union[str, Path]) -> str:
    """
    将相对路径转换为基于项目根目录的绝对路径
    
    Args:
        relative_path: 相对于项目根目录的路径
        
    Returns:
        str: 绝对路径
        
    Example:
        >>> abs_path = get_abs_path('data/train.csv')
        >>> print(abs_path)
        /home/user/UAV_PATH_PLANNING/data/train.csv
    """
    if os.path.isabs(relative_path):
        # 如果已经是绝对路径，直接返回
        return str(relative_path)
    
    project_root = get_project_root()
    abs_path = os.path.join(project_root, relative_path)
    return os.path.normpath(abs_path)


def ensure_dir(relative_path: Union[str, Path], create: bool = True) -> str:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        relative_path: 相对于项目根目录的路径
        create: 是否在目录不存在时创建，默认 True
        
    Returns:
        str: 目录的绝对路径
        
    Raises:
        FileNotFoundError: 当 create=False 且目录不存在时
        
    Example:
        >>> save_dir = ensure_dir('saved_models')
        >>> print(save_dir)
        /home/user/UAV_PATH_PLANNING/saved_models
    """
    abs_path = get_abs_path(relative_path)
    
    if not os.path.exists(abs_path):
        if create:
            os.makedirs(abs_path, exist_ok=True)
            print(f"Created directory: {abs_path}")
        else:
            raise FileNotFoundError(f"Directory not found: {abs_path}")
    
    return abs_path


def get_save_dir(subdir: str = '') -> str:
    """
    获取模型和数据保存目录
    
    Args:
        subdir: 子目录名称，例如 'models', 'logs', 'data'
        
    Returns:
        str: 保存目录的绝对路径（自动创建）
        
    Example:
        >>> model_dir = get_save_dir('models')
        >>> log_dir = get_save_dir('logs')
    """
    if subdir:
        save_path = os.path.join('saved_models', subdir)
    else:
        save_path = 'saved_models'
    
    return ensure_dir(save_path)


def get_model_path(model_name: str, subdir: str = '') -> str:
    """
    获取模型文件的完整路径
    
    Args:
        model_name: 模型文件名，例如 'actor_L1.pth'
        subdir: 子目录名称（可选）
        
    Returns:
        str: 模型文件的绝对路径
        
    Example:
        >>> model_path = get_model_path('Path_SAC_actor_L1.pth')
        >>> print(model_path)
        /home/user/UAV_PATH_PLANNING/saved_models/Path_SAC_actor_L1.pth
    """
    save_dir = get_save_dir(subdir)
    return os.path.join(save_dir, model_name)


def get_data_path(data_name: str, subdir: str = 'data') -> str:
    """
    获取数据文件的完整路径
    
    Args:
        data_name: 数据文件名，例如 'train.pkl'
        subdir: 子目录名称，默认 'data'
        
    Returns:
        str: 数据文件的绝对路径
        
    Example:
        >>> data_path = get_data_path('MASAC_new1.pkl')
        >>> print(data_path)
        /home/user/UAV_PATH_PLANNING/saved_models/data/MASAC_new1.pkl
    """
    save_dir = get_save_dir(subdir)
    return os.path.join(save_dir, data_name)


def get_resource_path(resource_type: str, resource_name: str = '') -> str:
    """
    获取资源文件路径（图片、音频等）
    
    Args:
        resource_type: 资源类型，'image' 或 'music'
        resource_name: 资源文件名（可选）
        
    Returns:
        str: 资源路径的绝对路径
        
    Example:
        >>> image_dir = get_resource_path('image')
        >>> leader_img = get_resource_path('image', 'leader.png')
    """
    resource_path = os.path.join('assignment', 'source', resource_type)
    abs_path = get_abs_path(resource_path)
    
    if resource_name:
        return os.path.join(abs_path, resource_name)
    return abs_path


def list_models(pattern: str = '*.pth', subdir: str = '') -> list:
    """
    列出保存目录中的所有模型文件
    
    Args:
        pattern: 文件匹配模式，默认 '*.pth'
        subdir: 子目录名称（可选）
        
    Returns:
        list: 模型文件路径列表
        
    Example:
        >>> models = list_models()
        >>> for model in models:
        ...     print(model)
    """
    import glob
    save_dir = get_save_dir(subdir)
    search_pattern = os.path.join(save_dir, pattern)
    return glob.glob(search_pattern)


# 常用路径快捷方式
PROJECT_ROOT = get_project_root()
SAVE_DIR = ensure_dir('saved_models')
LOG_DIR = ensure_dir(os.path.join('saved_models', 'logs'))
DATA_DIR = ensure_dir(os.path.join('saved_models', 'data'))


if __name__ == '__main__':
    # 测试代码
    print("=" * 50)
    print("路径管理工具测试")
    print("=" * 50)
    
    print(f"\n项目根目录: {get_project_root()}")
    print(f"保存目录: {SAVE_DIR}")
    print(f"日志目录: {LOG_DIR}")
    print(f"数据目录: {DATA_DIR}")
    
    print("\n相对路径转换测试:")
    test_path = get_abs_path('config/settings.json')
    print(f"  config/settings.json -> {test_path}")
    
    print("\n模型路径测试:")
    model_path = get_model_path('test_model.pth')
    print(f"  test_model.pth -> {model_path}")
    
    print("\n数据路径测试:")
    data_path = get_data_path('test_data.pkl')
    print(f"  test_data.pkl -> {data_path}")
    
    print("\n资源路径测试:")
    image_dir = get_resource_path('image')
    print(f"  image目录 -> {image_dir}")
    
    print("\n" + "=" * 50)

