"""
配置文件功能测试
验证配置加载、参数解析等功能是否正常
"""

import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils import load_config, get_train_params, get_test_params, print_config


def test_load_default_config():
    """测试加载默认配置"""
    print("\n" + "="*60)
    print("测试1：加载默认配置")
    print("="*60)
    
    try:
        config = load_config()
        print("✅ 默认配置加载成功")
        print_config(config)
        return True
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False


def test_load_custom_config():
    """测试加载自定义配置"""
    print("\n" + "="*60)
    print("测试2：加载多Follower配置")
    print("="*60)
    
    try:
        config_path = project_root / 'configs' / 'masac' / 'multi_follower.yaml'
        config = load_config(str(config_path))
        print("✅ 自定义配置加载成功")
        
        # 验证参数
        assert config['environment']['n_follower'] == 3, "Follower数量应为3"
        print(f"  验证：Follower数量 = {config['environment']['n_follower']} ✅")
        
        return True
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_train_params():
    """测试获取训练参数"""
    print("\n" + "="*60)
    print("测试3：获取训练参数")
    print("="*60)
    
    try:
        config = load_config()
        params = get_train_params(config)
        
        # 验证关键参数
        required_keys = [
            'n_leader', 'n_follower', 'ep_max', 'ep_len',
            'gamma', 'batch_size', 'hidden_dim', 'q_lr'
        ]
        
        for key in required_keys:
            assert key in params, f"缺少参数: {key}"
            print(f"  {key}: {params[key]} ✅")
        
        print("✅ 训练参数提取成功")
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_get_test_params():
    """测试获取测试参数"""
    print("\n" + "="*60)
    print("测试4：获取测试参数")
    print("="*60)
    
    try:
        config = load_config()
        params = get_test_params(config)
        
        # 验证关键参数
        required_keys = [
            'n_leader', 'n_follower', 'test_episode', 'ep_len',
            'hidden_dim', 'policy_lr'
        ]
        
        for key in required_keys:
            assert key in params, f"缺少参数: {key}"
            print(f"  {key}: {params[key]} ✅")
        
        print("✅ 测试参数提取成功")
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("配置文件功能测试")
    print("="*60)
    
    tests = [
        test_load_default_config,
        test_load_custom_config,
        test_get_train_params,
        test_get_test_params
    ]
    
    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("✅ 所有测试通过！配置文件功能正常工作")
        return 0
    else:
        print("❌ 部分测试失败，请检查配置文件")
        return 1


if __name__ == '__main__':
    sys.exit(main())

