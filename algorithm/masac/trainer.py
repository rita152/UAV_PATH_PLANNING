"""
MASAC 训练器
负责 SAC 算法的训练流程控制、模型保存和数据记录
"""
import torch
import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl
from utils import get_model_path, get_data_path, set_global_seed, get_episode_seed, print_seed_info
from .agent import Actor, Critic, Entropy
from .buffer import Memory
from .noise import Ornstein_Uhlenbeck_Noise


class Trainer:
    """
    MASAC 训练器
    
    职责分离设计：
    - __init__: 接受配置参数（环境实例、网络结构、算法超参数、智能体数量）
    - train: 接受训练参数（训练轮数、渲染等）
    
    这样设计的好处：
    1. 一个训练器对应一个环境，简洁明了
    2. 网络结构在初始化时确定（n_leader, n_follower决定网络维度）
    3. train方法只需要控制训练流程，不需要传递环境
    4. 适合大多数使用场景（固定环境训练）
    
    Args (配置参数):
        env: Gym环境实例
        n_leader: Leader数量（决定网络结构）
        n_follower: Follower数量（决定网络结构）
        state_dim: 状态维度
        action_dim: 动作维度
        max_action: 动作最大值
        min_action: 动作最小值
        hidden_dim: 网络隐藏层维度
        gamma: 折扣因子
        q_lr: Q网络学习率
        value_lr: Value网络学习率
        policy_lr: Policy学习率
        tau: 软更新系数
        batch_size: 批次大小
        memory_capacity: 经验池容量
        data_save_name: 数据保存文件名
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
                 gamma=0.9,
                 q_lr=3e-4,
                 value_lr=3e-3,
                 policy_lr=1e-3,
                 tau=1e-2,
                 batch_size=128,
                 memory_capacity=20000,
                 device='auto',
                 seed=42,
                 deterministic=False,
                 data_save_name='MASAC_new1.pkl'):
        
        # 环境实例
        self.env = env
        
        # 设备选择
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 打印设备信息
        self._print_device_info()
        
        # 随机种子管理
        self.base_seed = seed
        self.deterministic = deterministic
        
        # 设置初始全局种子
        set_global_seed(seed, deterministic)
        print_seed_info(seed, mode='train', deterministic=deterministic)
        
        # 智能体数量（决定网络结构）
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
        self.q_lr = q_lr
        self.value_lr = value_lr
        self.policy_lr = policy_lr
        self.tau = tau
        
        # 算法参数
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        
        # 数据保存路径
        self.data_path = get_data_path(data_save_name)
    
    def _print_device_info(self):
        """打印设备信息"""
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(self.device)
            gpu_memory = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            print(f"🚀 使用GPU训练: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print(f"💻 使用CPU训练")
    
    def _initialize_agents(self):
        """
        初始化智能体组件（Actor, Critic, Entropy）
        
        Returns:
            actors, critics, entropies: 智能体组件列表
        """
        actors = []
        critics = []
        entropies = []
        
        for i in range(self.n_agents):
            # 创建 Actor（移到GPU）
            actor = Actor(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                max_action=self.max_action,
                min_action=self.min_action,
                hidden_dim=self.hidden_dim,
                policy_lr=self.policy_lr,
                device=str(self.device)
            )
            actors.append(actor)
            
            # 创建 Critic（移到GPU）
            critic = Critic(
                state_dim=self.state_dim * self.n_agents,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                value_lr=self.value_lr,
                tau=self.tau,
                device=str(self.device)
            )
            critics.append(critic)
            
            # 创建 Entropy 调节器（移到GPU）
            entropy = Entropy(
                target_entropy=-0.1,
                lr=self.q_lr,
                device=str(self.device)
            )
            entropies.append(entropy)
        
        return actors, critics, entropies
    
    def _initialize_memory(self):
        """
        初始化经验回放缓冲区
        
        Returns:
            memory: 经验回放缓冲区
        """
        transition_dim = (2 * self.state_dim * self.n_agents + 
                         self.action_dim * self.n_agents + 
                         1 * self.n_agents)
        return Memory(capacity=self.memory_capacity, transition_dim=transition_dim)
    
    def _initialize_noise(self):
        """
        初始化 OU 噪声生成器
        
        Returns:
            noise: OU噪声生成器
        """
        return Ornstein_Uhlenbeck_Noise(
            mean=np.zeros((self.n_agents, self.action_dim)),
            sigma=0.1,
            theta=0.1,
            dt=1e-2
        )
    
    def _collect_experience(self, actors, observation, episode, ou_noise):
        """
        采集经验（选择动作并添加噪声）
        
        Args:
            actors: Actor列表
            observation: 当前观测
            episode: 当前轮数
            ou_noise: OU噪声生成器
            
        Returns:
            action: 执行的动作
        """
        action = np.zeros((self.n_agents, self.action_dim))
        
        # 选择动作
        for i in range(self.n_agents):
            action[i] = actors[i].choose_action(observation[i])
        
        # 前20轮添加 OU 噪声进行探索
        if episode <= 20:
            noise = ou_noise()
        else:
            noise = 0
        
        action = action + noise
        action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def _update_agents(self, actors, critics, entropies, memory):
        """
        更新智能体网络参数
        
        Args:
            actors: Actor列表
            critics: Critic列表
            entropies: Entropy列表
            memory: 经验回放缓冲区
        """
        # 从经验池采样（CPU数据）
        b_M = memory.sample(self.batch_size)
        b_s = b_M[:, :self.state_dim * self.n_agents]
        b_a = b_M[:, self.state_dim * self.n_agents : 
                     self.state_dim * self.n_agents + self.action_dim * self.n_agents]
        b_r = b_M[:, -self.state_dim * self.n_agents - 1 * self.n_agents : 
                     -self.state_dim * self.n_agents]
        b_s_ = b_M[:, -self.state_dim * self.n_agents:]
        
        # 转换为 Tensor 并移到 GPU
        b_s = torch.FloatTensor(b_s).to(self.device)
        b_a = torch.FloatTensor(b_a).to(self.device)
        b_r = torch.FloatTensor(b_r).to(self.device)
        b_s_ = torch.FloatTensor(b_s_).to(self.device)
        
        # 更新每个智能体
        for i in range(self.n_agents):
            # 计算目标 Q 值
            new_action, log_prob_ = actors[i].evaluate(
                b_s_[:, self.state_dim * i : self.state_dim * (i + 1)]
            )
            target_q1, target_q2 = critics[i].get_target_q_value(b_s_, new_action)
            target_q = b_r[:, i:(i + 1)] + self.gamma * (
                torch.min(target_q1, target_q2) - 
                entropies[i].alpha * log_prob_.sum(dim=-1, keepdim=True)
            )
            
            # 更新 Critic
            current_q1, current_q2 = critics[i].get_q_value(
                b_s, b_a[:, self.action_dim * i : self.action_dim * (i + 1)]
            )
            critics[i].update(current_q1, current_q2, target_q.detach())
            
            # 更新 Actor
            a, log_prob = actors[i].evaluate(
                b_s[:, self.state_dim * i : self.state_dim * (i + 1)]
            )
            q1, q2 = critics[i].get_q_value(b_s, a)
            q = torch.min(q1, q2)
            actor_loss = (entropies[i].alpha * log_prob.sum(dim=-1, keepdim=True) - q).mean()
            actors[i].update(actor_loss)
            
            # 更新 Entropy
            alpha_loss = -(entropies[i].log_alpha.exp() * (
                log_prob.sum(dim=-1, keepdim=True) + entropies[i].target_entropy
            ).detach()).mean()
            entropies[i].update(alpha_loss)
            
            # 软更新目标网络
            critics[i].soft_update()
    
    def _save_models(self, actors, episode):
        """
        保存模型参数（自动处理GPU/CPU）
        只保存两个文件：leader.pth 和 follower.pth
        follower.pth 包含所有Follower的独立权重
        
        Args:
            actors: Actor列表
            episode: 当前轮数
        """
        if episode % 20 == 0 and episode > 200:
            # 保存 Leader 模型（所有Leader的权重）
            leader_save_data = {}
            for i in range(self.n_leader):
                leader_save_data[f'leader_{i}'] = {
                    'net': actors[i].action_net.cpu().state_dict(),
                    'opt': actors[i].optimizer.state_dict()
                }
            torch.save(leader_save_data, get_model_path('leader.pth'))
            
            # 保存 Follower 模型（所有Follower的权重打包到一个文件）
            if self.n_follower > 0:
                follower_save_data = {}
                for j in range(self.n_follower):
                    follower_idx = self.n_leader + j
                    follower_save_data[f'follower_{j}'] = {
                        'net': actors[follower_idx].action_net.cpu().state_dict(),
                        'opt': actors[follower_idx].optimizer.state_dict()
                    }
                torch.save(follower_save_data, get_model_path('follower.pth'))
            
            # 保存后移回GPU
            for i in range(self.n_leader):
                actors[i].action_net.to(self.device)
            for j in range(self.n_follower):
                actors[self.n_leader + j].action_net.to(self.device)
    
    def _save_training_data(self, all_ep_r, all_ep_r0, all_ep_r1):
        """
        保存训练数据（奖励统计）
        
        Args:
            all_ep_r: 总奖励列表
            all_ep_r0: Leader奖励列表
            all_ep_r1: Follower奖励列表
            
        Returns:
            data: 统计数据字典
        """
        all_ep_r_mean = np.mean(np.array(all_ep_r), axis=0)
        all_ep_r_std = np.std(np.array(all_ep_r), axis=0)
        all_ep_L_mean = np.mean(np.array(all_ep_r0), axis=0)
        all_ep_L_std = np.std(np.array(all_ep_r0), axis=0)
        all_ep_F_mean = np.mean(np.array(all_ep_r1), axis=0)
        all_ep_F_std = np.std(np.array(all_ep_r1), axis=0)
        
        data = {
            "all_ep_r_mean": all_ep_r_mean,
            "all_ep_r_std": all_ep_r_std,
            "all_ep_L_mean": all_ep_L_mean,
            "all_ep_L_std": all_ep_L_std,
            "all_ep_F_mean": all_ep_F_mean,
            "all_ep_F_std": all_ep_F_std,
        }
        
        with open(self.data_path, 'wb') as f:
            pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)
        
        return data
    
    def _plot_results(self, data):
        """
        绘制训练曲线
        
        Args:
            data: 训练统计数据字典
        """
        all_ep_r_mean = data["all_ep_r_mean"]
        all_ep_r_std = data["all_ep_r_std"]
        all_ep_L_mean = data["all_ep_L_mean"]
        all_ep_L_std = data["all_ep_L_std"]
        all_ep_F_mean = data["all_ep_F_mean"]
        all_ep_F_std = data["all_ep_F_std"]
        
        # 计算置信区间
        all_ep_r_max = all_ep_r_mean + all_ep_r_std * 0.95
        all_ep_r_min = all_ep_r_mean - all_ep_r_std * 0.95
        all_ep_L_max = all_ep_L_mean + all_ep_L_std * 0.95
        all_ep_L_min = all_ep_L_mean - all_ep_L_std * 0.95
        all_ep_F_max = all_ep_F_mean + all_ep_F_std * 0.95
        all_ep_F_min = all_ep_F_mean - all_ep_F_std * 0.95
        
        # 绘制总奖励曲线
        plt.figure(1)
        plt.margins(x=0)
        plt.plot(np.arange(len(all_ep_r_mean)), all_ep_r_mean, 
                label='MASAC', color='#e75840')
        plt.fill_between(np.arange(len(all_ep_r_mean)), all_ep_r_max, all_ep_r_min, 
                         alpha=0.6, facecolor='#e75840')
        plt.xlabel('Episode')
        plt.ylabel('Total reward')
        
        # 绘制 Leader 奖励曲线
        plt.figure(2, figsize=(8, 4), dpi=150)
        plt.margins(x=0)
        plt.plot(np.arange(len(all_ep_L_mean)), all_ep_L_mean, 
                label='MASAC', color='#e75840')
        plt.fill_between(np.arange(len(all_ep_L_mean)), all_ep_L_max, all_ep_L_min, 
                         alpha=0.6, facecolor='#e75840')
        plt.xlabel('Episode')
        plt.ylabel('Leader reward')
        
        # 绘制 Follower 奖励曲线
        plt.figure(3, figsize=(8, 4), dpi=150)
        plt.margins(x=0)
        plt.plot(np.arange(len(all_ep_F_mean)), all_ep_F_mean, 
                label='MASAC', color='#e75840')
        plt.fill_between(np.arange(len(all_ep_F_mean)), all_ep_F_max, all_ep_F_min, 
                         alpha=0.6, facecolor='#e75840')
        plt.xlabel('Episode')
        plt.ylabel('Follower reward')
        plt.legend()
        plt.show()
    
    def train(self, ep_max=500, ep_len=1000, train_num=1, render=False):
        """
        执行完整的训练流程
        
        Args (训练参数):
            ep_max: 最大训练轮数
            ep_len: 每轮最大步数
            train_num: 训练次数（用于多次实验）
            render: 是否渲染
            
        Returns:
            data: 训练统计数据字典
        """
        print('SAC训练中...')
        
        # 初始化训练统计
        all_ep_r = [[] for _ in range(train_num)]
        all_ep_r0 = [[] for _ in range(train_num)]
        all_ep_r1 = [[] for _ in range(train_num)]
        
        for k in range(train_num):
            # 初始化组件
            actors, critics, entropies = self._initialize_agents()
            memory = self._initialize_memory()
            ou_noise = self._initialize_noise()
            
            # 打印表头
            print("\n" + "="*80)
            header_parts = ["Episode"]
            for i in range(self.n_leader):
                header_parts.append(f"Leader{i}")
            for j in range(self.n_follower):
                header_parts.append(f"Follower{j}")
            header_parts.append("Steps")
            header_parts.append("Status")
            print(" | ".join([f"{part:^12}" for part in header_parts]))
            print("="*80)
            
            # 训练循环
            for episode in range(ep_max):
                # 为每个episode设置不同的种子（确保可复现）
                episode_seed = get_episode_seed(self.base_seed, episode, mode='train')
                set_global_seed(episode_seed, self.deterministic)
                
                observation = self.env.reset()
                reward_total = 0
                
                # 为每个智能体单独统计奖励
                reward_leaders = [0.0] * self.n_leader
                reward_followers = [0.0] * self.n_follower
                
                for timestep in range(ep_len):
                    # 采集经验
                    action = self._collect_experience(actors, observation, episode, ou_noise)
                    
                    # 环境交互
                    observation_, reward, done, win, team_counter = self.env.step(action)
                    
                    # 存储经验
                    memory.store(observation.flatten(), action.flatten(), 
                               reward.flatten(), observation_.flatten())
                    
                    # 学习更新
                    if memory.counter > self.memory_capacity:
                        self._update_agents(actors, critics, entropies, memory)
                    
                    # 更新状态
                    observation = observation_
                    reward_total += reward.mean()
                    
                    # 分别累积每个智能体的奖励
                    for i in range(self.n_leader):
                        reward_leaders[i] += float(reward[i])
                    for j in range(self.n_follower):
                        reward_followers[j] += float(reward[self.n_leader + j])
                    
                    # 渲染
                    if render:
                        self.env.render()
                    
                    # 检查终止
                    if done:
                        break
                
                # 判断终止状态
                if done:
                    if win:
                        status = "✅ Success"
                        status_color = "\033[92m"  # 绿色
                    else:
                        status = "❌ Failure"
                        status_color = "\033[91m"  # 红色
                else:
                    status = "⏱️  Timeout"
                    status_color = "\033[93m"  # 黄色
                reset_color = "\033[0m"
                
                # 格式化输出
                output_parts = [f"{episode:^12d}"]
                
                # Leader奖励
                for i in range(self.n_leader):
                    output_parts.append(f"{reward_leaders[i]:^12.2f}")
                
                # Follower奖励
                for j in range(self.n_follower):
                    output_parts.append(f"{reward_followers[j]:^12.2f}")
                
                # 步数和状态
                output_parts.append(f"{timestep+1:^12d}")
                output_parts.append(f"{status_color}{status:^12}{reset_color}")
                
                print(" | ".join(output_parts))
                
                # 记录统计（保持向后兼容）
                all_ep_r[k].append(reward_total)
                all_ep_r0[k].append(reward_leaders[0])
                if self.n_follower > 0:
                    all_ep_r1[k].append(reward_followers[0])
                
                # 保存模型
                self._save_models(actors, episode)
        
        # 保存训练数据
        data = self._save_training_data(all_ep_r, all_ep_r0, all_ep_r1)
        
        # 绘制结果
        self._plot_results(data)
        
        # 关闭环境
        self.env.close()
        
        return data

