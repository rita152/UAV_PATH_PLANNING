"""
MASAC训练器
"""
import torch
import numpy as np
from .agent import Actor, Critic, Entropy
from .buffer import Memory
from .noise import Ornstein_Uhlenbeck_Noise
from utils.seed_utils import get_episode_seed, set_seed
from utils.device_utils import get_device


class MASACTrainer:
    """MASAC算法训练器"""
    
    def __init__(self, env, config):
        """
        初始化训练器
        
        Args:
            env: 环境实例
            config: 配置字典，包含以下参数:
                - n_leaders: 领导者数量
                - n_followers: 跟随者数量
                - state_dim: 状态维度
                - action_dim: 动作维度
                - max_action: 最大动作值
                - min_action: 最小动作值
                - gamma: 折扣因子
                - policy_lr: 策略学习率
                - value_lr: 价值学习率
                - q_lr: Q网络学习率
                - tau: 软更新系数
                - batch_size: 批次大小
                - memory_capacity: 记忆库容量
                - max_episodes: 最大训练轮数
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
        
        self.gamma = config.get('gamma', 0.9)
        self.batch_size = config.get('batch_size', 128)
        self.max_episodes = config.get('max_episodes', 500)
        self.max_steps = config.get('max_steps', 1000)
        
        # 获取计算设备（不打印，由train.py统一打印）
        device_config = config.get('device_config', {})
        if device_config:
            use_cuda = device_config.get('use_cuda', True)
            cuda_device = device_config.get('cuda_device', 0)
            if use_cuda and torch.cuda.is_available():
                self.device = torch.device(f'cuda:{cuda_device}')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        
        # 初始化智能体（传入设备）
        total_agents = self.n_leaders + self.n_followers
        self.actors = [
            Actor(
                self.state_dim,
                self.action_dim,
                self.max_action,
                self.min_action,
                lr=config.get('policy_lr', 1e-3),
                device=self.device
            ) for _ in range(total_agents)
        ]
        
        self.critics = [
            Critic(
                self.state_dim * total_agents,      # 所有智能体的状态维度
                self.action_dim * total_agents,     # ✅ 修复：所有智能体的动作维度
                lr=config.get('value_lr', 3e-3),
                tau=config.get('tau', 1e-2),
                device=self.device
            ) for _ in range(total_agents)
        ]
        
        self.entropies = [
            Entropy(
                target_entropy=config.get('target_entropy', -0.1),
                lr=config.get('q_lr', 3e-4),
                device=self.device
            ) for _ in range(total_agents)
        ]
        
        # 初始化记忆库
        memory_capacity = config.get('memory_capacity', 20000)
        memory_dims = 2 * self.state_dim * total_agents + \
                     self.action_dim * total_agents + total_agents
        self.memory = Memory(memory_capacity, memory_dims)
        
        # 初始化OU噪声
        self.noise = Ornstein_Uhlenbeck_Noise(
            mu=np.zeros((total_agents, self.action_dim))
        )
        
        # 获取种子配置
        self.seed_config = config.get('seed_config', {})
        self.use_seed = self.seed_config.get('enabled', False)
        self.base_seed = self.seed_config.get('base_seed', 42)
        self.use_episode_seed = self.seed_config.get('use_episode_seed', True)
        
        # 获取模型保存配置
        self.save_interval = config.get('save_interval', 20)
        self.save_threshold = config.get('save_threshold', 200)
        
        # ✅ 修复：获取噪声配置（不再硬编码）
        self.noise_episodes = config.get('noise_episodes', 20)
        
        # 获取logger（如果有）
        self.logger = config.get('logger', None)
    
    def train(self):
        """执行训练"""
        # ✅ 设置所有网络为训练模式
        for actor in self.actors:
            actor.action_net.train()
        for critic in self.critics:
            critic.critic_v.train()
            critic.target_critic_v.eval()  # 目标网络始终保持eval模式
        
        all_rewards = []
        leader_rewards = []      # 记录leader奖励
        follower_rewards = []    # 记录follower奖励
        
        for episode in range(self.max_episodes):
            # 为每轮训练设置不同的种子（不打印）
            if self.use_seed and self.use_episode_seed:
                episode_seed = get_episode_seed(self.base_seed, episode)
                set_seed(episode_seed)
                # ✅ 同时设置环境的种子
                state, info = self.env.reset(seed=episode_seed)
            else:
                state, info = self.env.reset()
            
            episode_reward = 0
            episode_leader_reward = 0
            episode_follower_reward = 0
            episode_follower_rewards_individual = [0.0] * self.n_followers
            
            for step in range(self.max_steps):
                # 选择动作
                actions = np.array([
                    self.actors[i].choose_action(state[i])
                    for i in range(len(self.actors))
                ])
                
                # ✅ 修复：使用配置的noise_episodes参数，不再硬编码
                if episode < self.noise_episodes:
                    noise = self.noise()
                else:
                    noise = 0
                actions = np.clip(actions + noise, self.min_action, self.max_action)
                
                # ✅ 执行动作（适配新接口：terminated, truncated, info）
                next_state, reward, terminated, truncated, info = self.env.step(actions)
                done = terminated or truncated
                
                # 存储经验
                self.memory.store_transition(
                    state.flatten(),
                    actions.flatten(),
                    reward.flatten(),
                    next_state.flatten()
                )
                
                # ✅ 修复：只要有batch_size个样本就开始学习，不需要等buffer填满
                # 这样可以大大提高训练效率（从20000步才开始学习改为128步就开始）
                if self.memory.memory_counter >= self.batch_size:
                    self._update_networks()
                
                state = next_state
                
                # 分离统计leader和follower的奖励
                episode_leader_reward += reward[0, 0]
                if self.n_followers > 0:
                    episode_follower_reward += reward[1:, 0].mean()
                    for f_idx in range(self.n_followers):
                        episode_follower_rewards_individual[f_idx] += reward[1 + f_idx, 0]
                episode_reward += reward.sum()
                
                if done:
                    break
            
            all_rewards.append(episode_reward)
            leader_rewards.append(episode_leader_reward)
            follower_rewards.append(episode_follower_reward)
            
            # 打印详细信息
            follower_str = ", ".join([f"F{i}: {episode_follower_rewards_individual[i]:.2f}" 
                                      for i in range(self.n_followers)])
            
            # ✅ 判断结束原因（使用info中的信息）
            if terminated:
                if info.get('win', False):
                    status = "✓ success"
                else:
                    status = "✗ failure"
            elif truncated:
                status = "⏱ timeout"
            else:
                status = "⏹ stopped"
            
            msg = (f"Episode {episode:3d}, Steps: {step+1:4d}, Total: {episode_reward:7.2f}, "
                   f"Leader: {episode_leader_reward:7.2f}, [{follower_str}] - {status}")
            
            # 使用logger或print
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
            
            # 保存模型（使用配置参数）
            if episode % self.save_interval == 0 and episode > self.save_threshold:
                self.save_models(self.config.get('output_dir', 'output'))
                msg = f"  💾 模型已保存 (episode {episode})"
                if self.logger:
                    self.logger.info(msg)
                else:
                    print(msg)
        
        # 返回所有奖励数据
        return {
            'total_rewards': all_rewards,
            'leader_rewards': leader_rewards,
            'follower_rewards': follower_rewards
        }
    
    def _update_networks(self):
        """更新所有网络"""
        # 采样批次数据
        batch = self.memory.sample(self.batch_size)
        
        total_agents = self.n_leaders + self.n_followers
        state_dim_total = self.state_dim * total_agents
        action_dim_total = self.action_dim * total_agents
        
        b_s = batch[:, :state_dim_total]
        b_a = batch[:, state_dim_total:state_dim_total + action_dim_total]
        b_r = batch[:, -state_dim_total - total_agents:-state_dim_total]
        b_s_ = batch[:, -state_dim_total:]
        
        # 将数据移到设备上（CPU或GPU）
        b_s = torch.FloatTensor(b_s).to(self.device)
        b_a = torch.FloatTensor(b_a).to(self.device)
        b_r = torch.FloatTensor(b_r).to(self.device)
        b_s_ = torch.FloatTensor(b_s_).to(self.device)
        
        # ✅ 优化：预先计算所有智能体的next_action和log_prob（避免重复计算）
        with torch.no_grad():
            next_actions_list = []
            log_probs_list = []
            
            for j in range(total_agents):
                next_a_j, log_prob_j = self.actors[j].evaluate(
                    b_s_[:, self.state_dim * j:self.state_dim * (j + 1)]
                )
                next_actions_list.append(next_a_j)
                log_probs_list.append(log_prob_j)
            
            next_actions_all = torch.cat(next_actions_list, dim=1)
        
        # 更新每个智能体
        for i in range(total_agents):
            # ===== 计算目标Q值 =====
            with torch.no_grad():
                # ✅ 使用预计算的next_actions和log_probs
                target_q1, target_q2 = self.critics[i].target_get_v(b_s_, next_actions_all)
                target_q = b_r[:, i:i+1] + self.gamma * (
                    torch.min(target_q1, target_q2) - 
                    self.entropies[i].alpha * log_probs_list[i]
                )
            
            # ===== 更新Critic =====
            # ✅ 修复：使用所有智能体的动作
            current_q1, current_q2 = self.critics[i].get_v(b_s, b_a)
            self.critics[i].learn(current_q1, current_q2, target_q)
            
            # ===== 更新Actor =====
            # 采样当前智能体i的新动作
            action_i, log_prob_i = self.actors[i].evaluate(
                b_s[:, self.state_dim * i:self.state_dim * (i + 1)]
            )
            
            # ✅ 修复：构建混合动作（确保不传递其他智能体的梯度）
            actions_mixed = b_a.detach().clone()
            actions_mixed[:, self.action_dim * i:self.action_dim * (i + 1)] = action_i
            
            # 使用混合动作计算Q值
            q1, q2 = self.critics[i].get_v(b_s, actions_mixed)
            q = torch.min(q1, q2)
            actor_loss = (self.entropies[i].alpha * log_prob_i - q).mean()
            self.actors[i].learn(actor_loss)
            
            # ===== 更新Entropy =====
            entropy_loss = -(
                self.entropies[i].log_alpha.exp() *
                (log_prob_i + self.entropies[i].target_entropy).detach()
            ).mean()
            self.entropies[i].learn(entropy_loss)
            # ✅ 移除alpha的手动更新（现在通过property自动计算）
            
            # 软更新目标网络
            self.critics[i].soft_update()
    
    def save_models(self, output_dir):
        """
        保存模型
        将leader和follower的权重分别保存为独立文件
        
        Args:
            output_dir: 输出目录
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存所有leader的权重到一个文件
        leader_models = []
        for i in range(self.n_leaders):
            leader_models.append({
                'net': self.actors[i].action_net.state_dict(),
                'opt': self.actors[i].optimizer.state_dict()
            })
        
        leader_path = os.path.join(output_dir, 'leader.pth')
        torch.save({
            'models': leader_models,
            'n_leaders': self.n_leaders
        }, leader_path)
        
        # 保存所有follower的权重到一个文件
        follower_models = []
        for i in range(self.n_followers):
            follower_idx = self.n_leaders + i
            follower_models.append({
                'net': self.actors[follower_idx].action_net.state_dict(),
                'opt': self.actors[follower_idx].optimizer.state_dict()
            })
        
        follower_path = os.path.join(output_dir, 'follower.pth')
        torch.save({
            'models': follower_models,
            'n_followers': self.n_followers
        }, follower_path)
        
        print(f"模型已保存到 {output_dir}")
        print(f"  - Leader模型: leader.pth ({self.n_leaders}个)")
        print(f"  - Follower模型: follower.pth ({self.n_followers}个)")
    
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
                self.actors[i].optimizer.load_state_dict(leader_models[i]['opt'])
            
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
                self.actors[follower_idx].optimizer.load_state_dict(follower_models[i]['opt'])
            
            print(f"✓ 已加载Follower模型: {follower_path} ({len(follower_models)}个)")
        else:
            print(f"⚠️ Follower模型文件不存在: {follower_path}")
        
        print(f"✓ 模型加载完成")

