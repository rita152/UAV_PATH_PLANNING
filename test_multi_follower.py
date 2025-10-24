"""
测试多follower配置功能
验证不同follower数量下环境是否正常工作
"""
import sys
import numpy as np
from utils.config_loader import ConfigLoader
from rl_env.path_env import RlGame


def test_follower_config(n_followers):
    """
    测试指定数量的follower配置
    
    Args:
        n_followers: follower数量
    """
    print(f"\n{'='*60}")
    print(f"测试 {n_followers} 个 Follower")
    print(f"{'='*60}")
    
    try:
        # 1. 创建环境
        print(f"\n1. 创建环境 (n_leaders=1, n_followers={n_followers})")
        env = RlGame(n=1, m=n_followers, render=False)
        print(f"   ✓ 环境创建成功")
        
        # 2. 重置环境
        print(f"\n2. 重置环境并获取初始状态")
        state = env.reset()
        print(f"   ✓ 环境重置成功")
        
        # 3. 验证状态维度
        expected_shape = (1 + n_followers, 7)
        actual_shape = state.shape
        print(f"\n3. 验证状态维度")
        print(f"   期望形状: {expected_shape}")
        print(f"   实际形状: {actual_shape}")
        
        if actual_shape == expected_shape:
            print(f"   ✓ 状态维度正确")
        else:
            print(f"   ✗ 状态维度错误！")
            return False
        
        # 4. 验证状态内容
        print(f"\n4. 验证状态内容")
        print(f"   Leader状态 (第1行):")
        print(f"      位置: ({state[0,0]:.3f}, {state[0,1]:.3f})")
        print(f"      速度: {state[0,2]:.3f}")
        print(f"      角度: {state[0,3]:.3f}")
        print(f"      目标: ({state[0,4]:.3f}, {state[0,5]:.3f})")
        
        for i in range(n_followers):
            print(f"   Follower{i}状态 (第{i+2}行):")
            print(f"      位置: ({state[i+1,0]:.3f}, {state[i+1,1]:.3f})")
            print(f"      速度: {state[i+1,2]:.3f}")
            print(f"      角度: {state[i+1,3]:.3f}")
            print(f"      跟随目标: ({state[i+1,4]:.3f}, {state[i+1,5]:.3f})")
        
        # 5. 验证状态值的有效性
        print(f"\n5. 验证状态值的有效性")
        if np.all(np.isfinite(state)):
            print(f"   ✓ 所有状态值有效（无NaN或Inf）")
        else:
            print(f"   ✗ 状态值包含NaN或Inf！")
            return False
        
        # 6. 测试step函数
        print(f"\n6. 测试环境交互（执行一步）")
        total_agents = 1 + n_followers
        actions = np.random.uniform(-1, 1, (total_agents, 2))
        print(f"   生成随机动作: shape={actions.shape}")
        
        next_state, reward, done, win, team_counter = env.step(actions)
        print(f"   ✓ Step执行成功")
        print(f"   下一状态形状: {next_state.shape}")
        print(f"   奖励形状: {reward.shape}")
        print(f"   完成标志: {done}")
        
        # 7. 验证奖励维度
        print(f"\n7. 验证奖励维度")
        expected_reward_shape = (total_agents, 1)
        actual_reward_shape = reward.shape
        print(f"   期望形状: {expected_reward_shape}")
        print(f"   实际形状: {actual_reward_shape}")
        
        if actual_reward_shape == expected_reward_shape:
            print(f"   ✓ 奖励维度正确")
        else:
            print(f"   ✗ 奖励维度错误！")
            return False
        
        print(f"\n{'='*60}")
        print(f"✅ {n_followers} 个Follower配置测试通过！")
        print(f"{'='*60}")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ 测试失败: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loader():
    """测试配置加载器"""
    print(f"\n{'='*60}")
    print(f"测试配置加载器")
    print(f"{'='*60}")
    
    try:
        # 测试默认配置
        print(f"\n1. 加载默认配置")
        loader = ConfigLoader('configs/masac/default.yaml')
        config = loader.load()
        
        n_followers = config['environment']['n_followers']
        print(f"   配置中的follower数量: {n_followers}")
        
        # 测试修改配置
        print(f"\n2. 测试配置修改")
        config['environment']['n_followers'] = 3
        print(f"   修改后的follower数量: {config['environment']['n_followers']}")
        
        print(f"\n✅ 配置加载器测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 配置加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("多Follower配置功能测试")
    print("="*60)
    
    results = []
    
    # 测试配置加载器
    print("\n【阶段1】配置加载器测试")
    results.append(("配置加载器", test_config_loader()))
    
    # 测试不同数量的follower
    print("\n【阶段2】环境初始化测试")
    
    test_cases = [1, 2, 3, 5]
    
    for n_followers in test_cases:
        result = test_follower_config(n_followers)
        results.append((f"{n_followers}个Follower", result))
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("="*60)
    if all_passed:
        print("\n🎉 所有测试通过！多Follower配置功能正常工作！")
        print("\n可以使用以下命令进行训练：")
        print("  python main_SAC.py --n_followers 1")
        print("  python main_SAC.py --n_followers 2")
        print("  python main_SAC.py --n_followers 3")
        print("  python main_SAC.py --n_followers 5")
        print()
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())

