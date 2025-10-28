"""
MASAC 训练器
负责 SAC 算法的训练流程控制、模型保存和数据记录
"""
import torch
import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl
import os
import sys
import re
import shutil
from datetime import datetime
from utils import set_global_seed, get_episode_seed, print_seed_info, get_project_root, load_config, set_env_vars
from .agent import Actor, Critic, Entropy
from .buffer import Memory
from .noise import Ornstein_Uhlenbeck_Noise
from rl_env.path_env import RlGame


class Logger:
    """
    同时输出到终端和文件的日志类
    实时写入，无缓冲
    终端保留颜色，文件去除ANSI颜色代码
    """
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', buffering=1)  # 行缓冲，实时写入
        # 编译正则表达式，用于去除ANSI颜色代码
        self.ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        
    def write(self, message):
        # 终端输出保留颜色
        self.terminal.write(message)
        # 文件输出去除颜色代码
        clean_message = self.ansi_escape.sub('', message)
        self.log.write(clean_message)
        self.log.flush()  # 强制刷新到磁盘
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


class Trainer:
    """
    MASAC 训练器
    
    简化设计：
    - 直接接受YAML配置文件路径
    - 自动创建环境和加载所有参数
    - 支持通过kwargs覆盖配置文件中的参数
    
    使用示例：
        # 使用默认配置
        trainer = Trainer(config="configs/masac/default.yaml")
        trainer.train()
        
        # 覆盖部分参数
        trainer = Trainer(config="configs/masac/default.yaml", 
                         ep_max=1000, device='cuda:1')
        trainer.train()
    
    Args:
        config: YAML配置文件路径
        **kwargs: 可选的参数覆盖（会覆盖配置文件中的对应参数）
    """
    def __init__(self, config, **kwargs):
        """
        初始化训练器
        
        Args:
            config: YAML配置文件路径
            **kwargs: 可选的参数覆盖
        """
        # 加载配置文件
        self.config_path = config
        cfg = load_config(config)
        
        # 使用kwargs覆盖配置
        for key, value in kwargs.items():
            if '.' in key:  # 支持嵌套参数，如 training.ep_max
                sections = key.split('.')
                target = cfg
                for section in sections[:-1]:
                    target = target[section]
                target[sections[-1]] = value
            else:
                # 自动查找并更新参数
                for section in cfg.values():
                    if isinstance(section, dict) and key in section:
                        section[key] = value
                        break
        
        # 设置环境变量
        set_env_vars(cfg)
        
        # 从配置中提取参数
        env_cfg = cfg['environment']
        train_cfg = cfg['training']
        net_cfg = cfg['network']
        output_cfg = cfg.get('output', {})
        
        # 创建环境
        self.env = RlGame(
            n=env_cfg['n_leader'],
            m=env_cfg['n_follower'],
            render=train_cfg.get('render', False)
        ).unwrapped
        
        # 从环境获取动作空间参数
        action_dim = self.env.action_space.shape[0]
        max_action = self.env.action_space.high[0]
        min_action = self.env.action_space.low[0]
        
        # 智能体数量（决定网络结构）
        self.n_leader = env_cfg['n_leader']
        self.n_follower = env_cfg['n_follower']
        self.n_agents = self.n_leader + self.n_follower
        
        # 状态和动作空间参数
        self.state_dim = env_cfg['state_dim']
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action
        
        # 网络参数
        self.hidden_dim = net_cfg['hidden_dim']
        self.q_lr = net_cfg['q_lr']
        self.value_lr = net_cfg['value_lr']
        self.policy_lr = net_cfg['policy_lr']
        self.tau = net_cfg['tau']
        
        # 训练算法参数
        self.gamma = train_cfg['gamma']
        self.batch_size = train_cfg['batch_size']
        self.memory_capacity = train_cfg['memory_capacity']
        
        # 训练参数（保存以便train()使用）
        self.ep_max = train_cfg['ep_max']
        self.ep_len = train_cfg['ep_len']
        self.train_num = train_cfg['train_num']
        self.render = train_cfg.get('render', False)
        
        # 设备选择
        device = cfg.get('device', 'auto')
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 打印设备信息
        self._print_device_info()
        
        # 随机种子管理
        seed_cfg = cfg.get('seed', {})
        self.base_seed = seed_cfg.get('base_seed', 42)
        self.deterministic = seed_cfg.get('deterministic', False)
        
        # 设置初始全局种子
        set_global_seed(self.base_seed, self.deterministic)
        print_seed_info(self.base_seed, mode='train', deterministic=self.deterministic)
        
        # 实验配置
        self.experiment_name = train_cfg.get('experiment_name', 'baseline')
        self.save_dir_prefix = train_cfg.get('save_dir_prefix', 'exp')
        
        # 输出配置
        self.verbose = output_cfg.get('verbose', True)
        self.log_interval = output_cfg.get('log_interval', 1)
        self.save_interval = output_cfg.get('save_interval', 20)
        
        # 创建独立的输出目录
        self.output_dir = self._create_output_dir(self.experiment_name, self.save_dir_prefix)
        print(f"📁 输出目录: {self.output_dir}")
        
        # 保存配置文件副本
        self._save_config(self.config_path)
        
        # 初始化日志系统
        self.logger = None
        self.original_stdout = None
        self._setup_logger()
    
    def _create_output_dir(self, experiment_name, save_dir_prefix):
        """
        创建独立的输出目录
        
        Args:
            experiment_name: 实验名称
            save_dir_prefix: 目录前缀
            
        Returns:
            输出目录的绝对路径
        """
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建目录名: exp_baseline_20251028_143022
        dir_name = f"{save_dir_prefix}_{experiment_name}_{timestamp}"
        
        # 完整路径（使用runs作为根目录）
        output_dir = os.path.join(get_project_root(), 'runs', dir_name)
        
        # 创建目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        return output_dir
    
    def _setup_logger(self):
        """
        设置日志系统，将输出重定向到文件和终端
        """
        log_file = os.path.join(self.output_dir, 'training.log')
        self.original_stdout = sys.stdout
        self.logger = Logger(log_file)
        sys.stdout = self.logger
        print(f"📝 日志文件: {log_file}")
        print(f"💡 训练输出将实时保存到日志文件")
    
    def _close_logger(self):
        """
        关闭日志系统，恢复标准输出
        """
        if self.logger is not None:
            sys.stdout = self.original_stdout
            self.logger.close()
            print(f"✅ 日志已保存: {os.path.join(self.output_dir, 'training.log')}")
    
    def _save_config(self, config_path):
        """
        保存配置文件副本到输出目录
        
        Args:
            config_path: 原始配置文件路径
        """
        if config_path and os.path.exists(config_path):
            dest_path = os.path.join(self.output_dir, 'config.yaml')
            shutil.copy(config_path, dest_path)
            print(f"✅ 配置文件已保存: {dest_path}")
    
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
        if episode % self.save_interval == 0 and episode > 200:
            # 保存 Leader 模型（所有Leader的权重）
            leader_save_data = {}
            for i in range(self.n_leader):
                leader_save_data[f'leader_{i}'] = {
                    'net': actors[i].action_net.cpu().state_dict(),
                    'opt': actors[i].optimizer.state_dict()
                }
            torch.save(leader_save_data, os.path.join(self.output_dir, 'leader.pth'))
            
            # 保存 Follower 模型（所有Follower的权重打包到一个文件）
            if self.n_follower > 0:
                follower_save_data = {}
                for j in range(self.n_follower):
                    follower_idx = self.n_leader + j
                    follower_save_data[f'follower_{j}'] = {
                        'net': actors[follower_idx].action_net.cpu().state_dict(),
                        'opt': actors[follower_idx].optimizer.state_dict()
                    }
                torch.save(follower_save_data, os.path.join(self.output_dir, 'follower.pth'))
            
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
        
        # 保存到输出目录
        data_path = os.path.join(self.output_dir, 'training_data.pkl')
        with open(data_path, 'wb') as f:
            pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)
        
        print(f"✅ 训练数据已保存: {data_path}")
        
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
        
        # 保存路径（使用输出目录）
        plot_dir = os.path.join(self.output_dir, 'plots')
        
        # 绘制总奖励曲线
        plt.figure(1, figsize=(8, 4), dpi=150)
        plt.margins(x=0)
        plt.plot(np.arange(len(all_ep_r_mean)), all_ep_r_mean, 
                label='MASAC', color='#e75840')
        plt.fill_between(np.arange(len(all_ep_r_mean)), all_ep_r_max, all_ep_r_min, 
                         alpha=0.6, facecolor='#e75840')
        plt.xlabel('Episode')
        plt.ylabel('Total reward')
        plt.title('Total Reward Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/total_reward.png', dpi=300)
        print(f"✅ 总奖励曲线已保存: {plot_dir}/total_reward.png")
        
        # 绘制 Leader 奖励曲线
        plt.figure(2, figsize=(8, 4), dpi=150)
        plt.margins(x=0)
        plt.plot(np.arange(len(all_ep_L_mean)), all_ep_L_mean, 
                label='MASAC', color='#e75840')
        plt.fill_between(np.arange(len(all_ep_L_mean)), all_ep_L_max, all_ep_L_min, 
                         alpha=0.6, facecolor='#e75840')
        plt.xlabel('Episode')
        plt.ylabel('Leader reward')
        plt.title('Leader Reward Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/leader_reward.png', dpi=300)
        print(f"✅ Leader奖励曲线已保存: {plot_dir}/leader_reward.png")
        
        # 绘制 Follower 奖励曲线
        plt.figure(3, figsize=(8, 4), dpi=150)
        plt.margins(x=0)
        plt.plot(np.arange(len(all_ep_F_mean)), all_ep_F_mean, 
                label='MASAC', color='#e75840')
        plt.fill_between(np.arange(len(all_ep_F_mean)), all_ep_F_max, all_ep_F_min, 
                         alpha=0.6, facecolor='#e75840')
        plt.xlabel('Episode')
        plt.ylabel('Follower reward')
        plt.title('Follower Reward Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/follower_reward.png', dpi=300)
        print(f"✅ Follower奖励曲线已保存: {plot_dir}/follower_reward.png")
    
    def train(self, ep_max=None, ep_len=None, train_num=None, render=None):
        """
        执行完整的训练流程
        
        Args (可选，用于临时覆盖配置):
            ep_max: 最大训练轮数（默认使用配置文件中的值）
            ep_len: 每轮最大步数（默认使用配置文件中的值）
            train_num: 训练次数（默认使用配置文件中的值）
            render: 是否渲染（默认使用配置文件中的值）
            
        Returns:
            data: 训练统计数据字典
        """
        # 使用配置文件中的参数作为默认值
        ep_max = ep_max if ep_max is not None else self.ep_max
        ep_len = ep_len if ep_len is not None else self.ep_len
        train_num = train_num if train_num is not None else self.train_num
        render = render if render is not None else self.render
        
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
            # Leader列（只有1个leader，直接使用"Leader"）
            header_parts.append("Leader")
            # Follower列（根据数量添加）
            for j in range(self.n_follower):
                if self.n_follower == 1:
                    header_parts.append("Follower")
                else:
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
        
        # 关闭日志
        self._close_logger()
        
        # 关闭环境
        self.env.close()
        
        return data

