"""
MASAC测试器
"""
import torch
import numpy as np
from .agent import Actor
from utils.device_utils import get_device
from utils.normalization import get_normalizer  # ✅ 导入归一化工具


class MASACTester:
    """MASAC算法测试器"""
    
    def __init__(self, env, config):
        """
        初始化测试器
        
        Args:
            env: 环境实例
            config: 配置字典，包含以下参数:
                - n_leaders: 领导者数量
                - n_followers: 跟随者数量
                - state_dim: 状态维度
                - action_dim: 动作维度
                - max_action: 最大动作值
                - min_action: 最小动作值
                - test_episodes: 测试轮数
                - max_steps: 每轮最大步数
        """
        self.env = env
        self.config = config
        
        # 提取配置参数
        self.n_leaders = config.get('n_leaders', 1)
        self.n_followers = config.get('n_followers', 1)
        self.state_dim = config.get('state_dim', 7)
        self.action_dim = config.get('action_dim', 2)
        self.max_action = config.get('max_action', 1.0)
        self.min_action = config.get('min_action', -1.0)
        
        self.test_episodes = config.get('test_episodes', 100)
        self.max_steps = config.get('max_steps', 1000)
        
        # 获取计算设备
        device_config = config.get('device_config', {})
        self.device = get_device(device_config) if device_config else torch.device('cpu')
        
        # 初始化Actor（仅用于测试，传入设备）
        total_agents = self.n_leaders + self.n_followers
        self.actors = [
            Actor(
                self.state_dim,
                self.action_dim,
                self.max_action,
                self.min_action,
                device=self.device
            ) for _ in range(total_agents)
        ]
    
    def load_models(self, output_dir):
        """
        加载模型
        从leader.pth和follower.pth分别加载权重
        
        Args:
            output_dir: 模型目录
        """
        import os
        
        # 加载leader模型
        leader_path = os.path.join(output_dir, 'leader.pth')
        if os.path.exists(leader_path):
            checkpoint = torch.load(leader_path, map_location=self.device)
            leader_models = checkpoint['models']
            saved_n_leaders = checkpoint['n_leaders']
            
            if saved_n_leaders != self.n_leaders:
                print(f"⚠️ 警告: 模型中的leader数量({saved_n_leaders})与当前配置({self.n_leaders})不匹配")
            
            for i in range(min(self.n_leaders, len(leader_models))):
                self.actors[i].action_net.load_state_dict(leader_models[i]['net'])
            
            print(f"✓ 已加载Leader模型: {leader_path} ({len(leader_models)}个)")
        else:
            print(f"⚠️ Leader模型文件不存在: {leader_path}")
        
        # 加载follower模型
        follower_path = os.path.join(output_dir, 'follower.pth')
        if os.path.exists(follower_path):
            checkpoint = torch.load(follower_path, map_location=self.device)
            follower_models = checkpoint['models']
            saved_n_followers = checkpoint['n_followers']
            
            if saved_n_followers != self.n_followers:
                print(f"⚠️ 警告: 模型中的follower数量({saved_n_followers})与当前配置({self.n_followers})不匹配")
            
            for i in range(min(self.n_followers, len(follower_models))):
                follower_idx = self.n_leaders + i
                self.actors[follower_idx].action_net.load_state_dict(follower_models[i]['net'])
            
            print(f"✓ 已加载Follower模型: {follower_path} ({len(follower_models)}个)")
        else:
            print(f"⚠️ Follower模型文件不存在: {follower_path}")
        
        print(f"✓ 模型加载完成")
    
    def test(self, render=False):
        """
        执行测试
        
        Args:
            render: 是否渲染环境
            
        Returns:
            测试结果字典
        """
        # ✅ 设置所有网络为评估模式
        for actor in self.actors:
            actor.action_net.eval()
        
        print(f"开始测试，共 {self.test_episodes} 轮...")
        
        win_count = 0
        total_rewards = []
        leader_rewards = []
        follower_rewards = []
        total_steps = []
        formation_keeping_rates = []
        all_integral_V = []
        all_integral_U = []
        
        for episode in range(self.test_episodes):
            # ✅ 适配新的reset接口
            state, info = self.env.reset()
            episode_reward = 0
            leader_episode_reward = 0
            follower_episode_reward = 0
            step = 0
            integral_V = 0
            integral_U = 0
            episode_team_counter = 0
            
            for step in range(self.max_steps):
                # 选择动作（确定性策略 - 用于测试）
                actions = np.zeros((len(self.actors), self.action_dim))
                for i in range(self.n_leaders):
                    actions[i] = self.actors[i].choose_action(state[i], deterministic=True)
                for i in range(self.n_followers):
                    actions[i + self.n_leaders] = self.actors[i + self.n_leaders].choose_action(
                        state[i + self.n_leaders], deterministic=True
                    )
                
                # ✅ 适配新的step接口
                next_state, reward, terminated, truncated, info = self.env.step(actions)
                done = terminated or truncated
                
                # ✅ 修复：使用归一化工具类进行反归一化
                normalizer = get_normalizer()
                current_speed_norm = state[0][2]  # 归一化的速度
                current_speed = normalizer.denormalize_speed(current_speed_norm)
                integral_V += current_speed
                
                # 能量损耗（所有智能体的动作）
                integral_U += np.abs(actions).sum()
                
                # 分离奖励统计
                leader_episode_reward += reward[0, 0]
                if self.n_followers > 0:
                    follower_episode_reward += reward[1:, 0].mean()
                episode_reward += reward.sum()
                
                # 累计编队计数（从info中获取）
                episode_team_counter += info.get('formation_count', 0)
                
                state = next_state
                
                if render:
                    self.env.render()
                
                if done:
                    if info.get('win', False):
                        win_count += 1
                    break
            
            # 记录统计信息
            total_rewards.append(episode_reward)
            leader_rewards.append(leader_episode_reward)
            follower_rewards.append(follower_episode_reward)
            total_steps.append(step + 1)
            all_integral_V.append(integral_V)
            all_integral_U.append(integral_U)
            
            # 编队保持率计算
            if step > 0 and self.n_followers > 0:
                formation_rate = episode_team_counter / ((step + 1) * self.n_followers)
                formation_keeping_rates.append(formation_rate)
            
            # ✅ 更新打印信息，使用info判断
            win_status = '是' if info.get('win', False) else '否'
            print(f"测试轮次 {episode + 1}/{self.test_episodes}, "
                  f"奖励: {episode_reward:.2f}, 步数: {step + 1}, "
                  f"胜利: {win_status}")
        
        # 计算统计结果
        results = {
            'success_rate': win_count / self.test_episodes,
            'total_avg_reward': np.mean(total_rewards),
            'leader_avg_reward': np.mean(leader_rewards),
            'follower_avg_reward': np.mean(follower_rewards),
            'std_reward': np.std(total_rewards),
            'avg_steps': np.mean(total_steps),
            'avg_formation_keeping': np.mean(formation_keeping_rates) if formation_keeping_rates else 0,
            'avg_flight_distance': np.mean(all_integral_V),
            'avg_energy_consumption': np.mean(all_integral_U),
            'total_episodes': self.test_episodes,
            'win_count': win_count
        }
        
        # 打印测试结果
        print("\n" + "="*50)
        print("测试结果汇总")
        print("="*50)
        print(f"任务完成率: {results['success_rate']*100:.2f}%")
        print(f"总平均奖励: {results['total_avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  - Leader奖励: {results['leader_avg_reward']:.2f}")
        print(f"  - Follower奖励: {results['follower_avg_reward']:.2f}")
        print(f"平均飞行时间: {results['avg_steps']:.2f}")
        print(f"平均飞行路程: {results['avg_flight_distance']:.2f}")
        print(f"平均能量损耗: {results['avg_energy_consumption']:.2f}")
        print(f"平均编队保持率: {results['avg_formation_keeping']*100:.2f}%")
        print(f"成功次数: {results['win_count']}/{results['total_episodes']}")
        print("="*50)
        
        return results
    
    def test_single_episode(self, render=True):
        """
        测试单个回合（可视化）
        
        Args:
            render: 是否渲染
            
        Returns:
            回合信息字典
        """
        # ✅ 设置所有网络为评估模式
        for actor in self.actors:
            actor.action_net.eval()
        
        print("开始单回合测试...")
        
        # ✅ 适配新的reset接口
        state, info = self.env.reset()
        episode_reward = 0
        trajectory = []
        
        for step in range(self.max_steps):
            # 选择动作（确定性策略 - 用于测试）
            actions = np.zeros((len(self.actors), self.action_dim))
            for i in range(len(self.actors)):
                actions[i] = self.actors[i].choose_action(state[i], deterministic=True)
            
            # ✅ 适配新的step接口
            next_state, reward, terminated, truncated, info = self.env.step(actions)
            done = terminated or truncated
            
            # 记录轨迹
            trajectory.append({
                'step': step,
                'state': state.copy(),
                'action': actions.copy(),
                'reward': reward.copy(),
                'next_state': next_state.copy(),
                'info': info.copy()
            })
            
            episode_reward += reward.mean()
            state = next_state
            
            if render:
                self.env.render()
            
            if done:
                break
        
        results = {
            'total_reward': episode_reward,
            'steps': step + 1,
            'win': info.get('win', False),
            'terminated': info.get('terminated', False),
            'truncated': info.get('truncated', False),
            'trajectory': trajectory
        }
        
        win_status = '是' if results['win'] else '否'
        print(f"单回合测试完成: 奖励={episode_reward:.2f}, "
              f"步数={step + 1}, 胜利={win_status}")
        
        return results

