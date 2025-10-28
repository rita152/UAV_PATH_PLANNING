"""
MASAC 测试器
负责加载训练好的模型并进行性能评估
"""
import torch
import numpy as np
from utils import get_model_path, set_global_seed, get_episode_seed, print_seed_info
from .agent import Actor


class Tester:
    """
    MASAC 测试器
    
    职责分离设计：
    - __init__: 接受配置参数（环境实例、网络结构、模型路径、智能体数量）
    - test: 接受测试参数（测试轮数、渲染等）
    
    这样设计的好处：
    1. 一个测试器对应一个环境，简洁明了
    2. 网络结构在初始化时确定（n_leader, n_follower决定测试流程）
    3. test方法只需要控制测试流程，不需要传递环境
    4. 适合大多数使用场景（固定环境测试）
    
    Args (配置参数):
        env: Gym环境实例
        n_leader: Leader数量（决定测试流程）
        n_follower: Follower数量（决定测试流程）
        state_dim: 状态维度
        action_dim: 动作维度
        max_action: 动作最大值
        min_action: 动作最小值
        hidden_dim: 网络隐藏层维度
        policy_lr: Policy学习率
        leader_model_path: Leader模型路径（默认自动获取）
        follower_model_path: Follower模型路径（默认自动获取）
    """
    def __init__(self,
                 env,
                 n_leader,
                 n_follower,
                 state_dim,
                 action_dim,
                 max_action,
                 min_action,
                 hidden_dim=256,
                 policy_lr=1e-3,
                 device='auto',
                 seed=42,
                 leader_model_path=None,
                 follower_model_path=None):
        
        # 环境实例
        self.env = env
        
        # 设备选择
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 打印设备信息
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(self.device)
            print(f"🚀 使用GPU测试: {gpu_name}")
        else:
            print(f"💻 使用CPU测试")
        
        # 随机种子管理（测试使用不同的种子空间）
        self.base_seed = seed
        
        # 设置初始全局种子
        set_global_seed(seed, deterministic=False)
        print_seed_info(seed, mode='test', deterministic=False)
        
        # 智能体数量
        self.n_leader = n_leader
        self.n_follower = n_follower
        self.n_agents = n_leader + n_follower
        
        # 状态和动作空间参数
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action
        
        # 网络参数
        self.hidden_dim = hidden_dim
        self.policy_lr = policy_lr
        
        # 模型路径（所有Follower共享同一个权重文件）
        self.leader_model_path = leader_model_path or get_model_path('leader.pth')
        self.follower_model_path = follower_model_path or get_model_path('follower.pth')
    
    def _load_actors(self):
        """
        加载所有智能体的模型
        leader.pth 包含所有Leader的权重
        follower.pth 包含所有Follower的权重
        
        Returns:
            actors: Actor列表（每个智能体独立）
        """
        actors = []
        
        # 加载Leader模型
        leader_checkpoint = torch.load(self.leader_model_path, map_location=self.device)
        
        for i in range(self.n_leader):
            actor = Actor(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                max_action=self.max_action,
                min_action=self.min_action,
                hidden_dim=self.hidden_dim,
                policy_lr=self.policy_lr,
                device=str(self.device)
            )
            # 加载对应Leader的权重
            actor.action_net.load_state_dict(leader_checkpoint[f'leader_{i}']['net'])
            actors.append(actor)
        
        # 加载Follower模型
        if self.n_follower > 0:
            follower_checkpoint = torch.load(self.follower_model_path, map_location=self.device)
            
            for j in range(self.n_follower):
                actor = Actor(
                    state_dim=self.state_dim,
                    action_dim=self.action_dim,
                    max_action=self.max_action,
                    min_action=self.min_action,
                    hidden_dim=self.hidden_dim,
                    policy_lr=self.policy_lr,
                    device=str(self.device)
                )
                # 加载对应Follower的权重
                actor.action_net.load_state_dict(follower_checkpoint[f'follower_{j}']['net'])
                actors.append(actor)
        
        return actors
    
    def _select_actions(self, actors, state):
        """
        选择动作（使用训练好的策略，无探索噪声）
        每个智能体使用自己独立的权重
        
        Args:
            actors: Actor列表（每个智能体独立）
            state: 当前状态
            
        Returns:
            action: 动作数组
        """
        action = np.zeros((self.n_agents, self.action_dim))
        
        # 每个智能体使用自己的策略网络选择动作
        for i in range(self.n_agents):
            action[i] = actors[i].choose_action(state[i])
        
        return action
    
    def test(self, ep_len=1000, test_episode=100, render=False):
        """
        执行完整的测试流程
        
        Args (测试参数):
            ep_len: 每轮最大步数
            test_episode: 测试轮数
            render: 是否渲染
            
        Returns:
            results: 测试结果字典
        """
        print('SAC测试中...')
        
        # 加载所有智能体的模型（每个智能体独立权重）
        actors = self._load_actors()
        
        # 初始化统计变量
        win_times = 0
        average_FKR = 0
        average_timestep = 0
        average_integral_V = 0
        average_integral_U = 0
        all_ep_V = []
        all_ep_U = []
        all_ep_T = []
        all_ep_F = []
        
        # 测试循环
        for j in range(test_episode):
            # 为每个episode设置不同的种子（测试种子空间）
            episode_seed = get_episode_seed(self.base_seed, j, mode='test')
            set_global_seed(episode_seed, deterministic=False)
            
            state = self.env.reset()
            total_rewards = 0
            integral_V = 0
            integral_U = 0
            v, v1 = [], []
            
            for timestep in range(ep_len):
                # 选择动作（每个智能体使用自己的权重）
                action = self._select_actions(actors, state)
                
                # 执行动作
                new_state, reward, done, win, team_counter = self.env.step(action)
                
                # 记录胜利
                if win:
                    win_times += 1
                
                # 记录数据
                v.append(state[0][2] * 30)
                v1.append(state[1][2] * 30)
                integral_V += state[0][2]
                integral_U += abs(action[0]).sum()
                total_rewards += reward.mean()
                
                # 更新状态
                state = new_state
                
                # 渲染
                if render:
                    self.env.render()
                
                # 检查终止
                if done:
                    break
            
            # 更新统计
            FKR = team_counter / timestep if timestep > 0 else 0
            average_FKR += FKR
            average_timestep += timestep
            average_integral_V += integral_V
            average_integral_U += integral_U
            all_ep_V.append(integral_V)
            all_ep_U.append(integral_U)
            all_ep_T.append(timestep)
            all_ep_F.append(FKR)
            
            print("Score", total_rewards)
        
        # 打印结果
        print('任务完成率', win_times / test_episode)
        print('平均最大编队保持率', average_FKR / test_episode)
        print('平均最短飞行时间', average_timestep / test_episode)
        print('平均最短飞行路程', average_integral_V / test_episode)
        print('平均最小能量损耗', average_integral_U / test_episode)
        
        # 关闭环境
        self.env.close()
        
        # 返回结果
        results = {
            'win_rate': win_times / test_episode,
            'average_FKR': average_FKR / test_episode,
            'average_timestep': average_timestep / test_episode,
            'average_integral_V': average_integral_V / test_episode,
            'average_integral_U': average_integral_U / test_episode,
            'all_ep_V': all_ep_V,
            'all_ep_U': all_ep_U,
            'all_ep_T': all_ep_T,
            'all_ep_F': all_ep_F,
        }
        
        return results

