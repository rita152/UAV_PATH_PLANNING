# 🗺️ GNN-Transformer架构实施路线图

**项目**: UAV路径规划系统架构升级  
**实施计划日期**: 2025-10-31  
**预计完成时间**: 2-6周（根据选择方案）  
**目标**: 编队率95%+，任务完成率95%+

---

## 📋 执行摘要

本文档提供GNN-Transformer混合架构的详细实施计划，分为3个阶段，每个阶段包含明确的实施内容、验收标准和回滚策略。

**实施策略**: 渐进式部署，风险可控，每阶段独立验证

**核心原则**:
1. 🎯 **小步快跑**：每个阶段2-3周，快速验证
2. 🔄 **可回滚**：任何阶段失败都可以回退
3. 📊 **数据驱动**：基于性能指标决定是否进入下一阶段
4. 🧪 **充分测试**：每阶段50-200 episodes验证

---

## 目录

1. [总体规划](#1-总体规划)
2. [阶段1：Het-GAT基础实施](#2-阶段1het-gat基础实施)
3. [阶段2：GNN-Transformer混合](#3-阶段2gnn-transformer混合)
4. [阶段3：性能优化与扩展](#4-阶段3性能优化与扩展)
5. [验收标准体系](#5-验收标准体系)
6. [风险管理与回滚](#6-风险管理与回滚)
7. [资源需求评估](#7-资源需求评估)

---

## 1. 总体规划

### 1.1 三阶段路线图

```
┌────────────────────────────────────────────────────────┐
│  阶段1: Het-GAT基础实施 (Week 1-2)                    │
├────────────────────────────────────────────────────────┤
│  实施内容：                                            │
│  ✅ 图数据结构设计                                    │
│  ✅ 异构GAT Actor实现                                 │
│  ✅ 异构GAT Critic实现                                │
│  ✅ 图batch处理                                       │
│                                                        │
│  验收标准：                                            │
│  🎯 编队率 ≥ 85%                                      │
│  🎯 完成率 ≥ 85%                                      │
│  🎯 TIMEOUT率 ≤ 7%                                    │
│  🎯 训练稳定（无NaN/崩溃）                            │
│                                                        │
│  交付物：                                              │
│  📦 gnn_actor.py, gnn_critic.py                       │
│  📦 graph_builder.py                                  │
│  📦 训练脚本（train_gnn.py）                          │
│  📦 验证报告（50 episodes）                           │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  阶段2: GNN-Transformer混合 (Week 3-5)                │
├────────────────────────────────────────────────────────┤
│  实施内容：                                            │
│  ✅ Transformer编码器实现                             │
│  ✅ 空间位置编码                                      │
│  ✅ GNN-Trans融合层                                   │
│  ✅ 端到端训练                                        │
│                                                        │
│  验收标准：                                            │
│  🎯 编队率 ≥ 95%                                      │
│  🎯 完成率 ≥ 95%                                      │
│  🎯 TIMEOUT率 ≤ 3%                                    │
│  🎯 性能超越阶段1                                     │
│                                                        │
│  交付物：                                              │
│  📦 transformer_layer.py                              │
│  📦 hybrid_model.py                                   │
│  📦 完整训练（500 episodes）                          │
│  📦 性能对比报告                                      │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  阶段3: 性能优化与扩展 (Week 6-8, 可选)              │
├────────────────────────────────────────────────────────┤
│  实施内容：                                            │
│  ✅ 多障碍物支持                                      │
│  ✅ 动态环境适应                                      │
│  ✅ 超参数自动优化                                    │
│  ✅ 模型压缩与加速                                    │
│                                                        │
│  验收标准：                                            │
│  🎯 支持10+障碍物                                     │
│  🎯 支持10+follower                                   │
│  🎯 推理速度 > 30 FPS                                 │
│  🎯 模型大小 < 10MB                                   │
│                                                        │
│  交付物：                                              │
│  📦 扩展环境（multi_obstacle_env.py）                 │
│  📦 压缩模型（pruned_model.pth）                      │
│  📦 论文材料                                          │
└────────────────────────────────────────────────────────┘
```

### 1.2 时间线甘特图

```
Week │ 1   2   3   4   5   6   7   8
─────┼────────────────────────────────
阶段1 │ ████████
     │ │  │  │
     │ 实验调试验收
     │
阶段2 │         ████████████
     │         │     │   │  │
     │         实现  集成 训练 验收
     │
阶段3 │                     ████████████
     │                     (可选研究)
```

### 1.3 决策检查点

```
Checkpoint 1 (Week 2末):
├─ 阶段1是否成功？
├─ YES → 进入阶段2
└─ NO → 分析原因，调整或回退

Checkpoint 2 (Week 5末):
├─ 阶段2是否成功？
├─ YES → 决定是否进入阶段3
└─ NO → 回退到阶段1或FC

Checkpoint 3 (Week 8末):
├─ 最终性能是否满足要求？
├─ YES → 部署生产
└─ NO → 使用最佳可用版本
```

---

## 2. 阶段1：Het-GAT基础实施

### 2.1 实施内容清单

#### **Task 1.1: 环境准备与依赖安装**

**时间**: 0.5天

**实施步骤**：
```bash
# 1. 安装PyTorch Geometric
pip install torch-geometric

# 2. 验证安装
python -c "import torch_geometric; print(torch_geometric.__version__)"

# 3. 安装可视化工具（可选）
pip install networkx matplotlib

# 4. 更新requirements.txt
echo "torch-geometric>=2.5.0" >> requirements.txt
echo "torch-scatter>=2.1.0" >> requirements.txt
echo "torch-sparse>=0.6.0" >> requirements.txt
```

**验收标准**：
- ✅ 成功导入`torch_geometric`
- ✅ 无版本冲突
- ✅ GPU支持正常

---

#### **Task 1.2: 图数据结构设计**

**时间**: 1天

**实施步骤**：

**文件**: `rl_env/graph_builder.py`（新建，约200行）

```python
"""
图数据构建器
功能：将环境状态转换为PyTorch Geometric图数据
"""

import torch
import numpy as np
from torch_geometric.data import Data
import math

class FormationGraphBuilder:
    """编队图构建器"""
    
    def __init__(self, formation_radius=50, coordination_radius=100):
        self.formation_radius = formation_radius
        self.coordination_radius = coordination_radius
        
        # 节点类型映射
        self.NODE_TYPES = {
            'leader': 0,
            'follower': 1,
            'goal': 2,
            'obstacle': 3
        }
    
    def build_graph(self, env_state, leader, followers, goal, obstacles):
        """
        从环境状态构建图
        
        Args:
            env_state: 当前环境状态
            leader: Leader对象
            followers: Follower对象列表
            goal: Goal对象
            obstacles: Obstacle对象列表
        
        Returns:
            PyG Data对象
        """
        # === 构建节点 ===
        nodes = []
        node_types = []
        positions = []
        
        # Leader节点
        leader_features = self._extract_leader_features(leader, goal)
        nodes.append(leader_features)
        node_types.append(self.NODE_TYPES['leader'])
        positions.append([leader.posx, leader.posy])
        
        # Follower节点
        for follower in followers:
            follower_features = self._extract_follower_features(follower, leader)
            nodes.append(follower_features)
            node_types.append(self.NODE_TYPES['follower'])
            positions.append([follower.posx, follower.posy])
        
        # Goal节点
        goal_features = self._extract_goal_features(goal)
        nodes.append(goal_features)
        node_types.append(self.NODE_TYPES['goal'])
        positions.append([goal.init_x, goal.init_y])
        
        # Obstacle节点
        for obs in obstacles:
            obs_features = self._extract_obstacle_features(obs)
            nodes.append(obs_features)
            node_types.append(self.NODE_TYPES['obstacle'])
            positions.append([obs.init_x, obs.init_y])
        
        # === 构建边 ===
        edge_index, edge_attr, edge_types = self._build_edges(
            positions, node_types, len(followers)
        )
        
        # === 构建PyG Data ===
        graph = Data(
            x=torch.tensor(nodes, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            node_type=torch.tensor(node_types, dtype=torch.long),
            pos=torch.tensor(positions, dtype=torch.float32)
        )
        
        return graph
    
    def _extract_leader_features(self, leader, goal):
        """提取Leader节点特征（8维）"""
        # 归一化
        x_norm = leader.posx / 1000.0
        y_norm = leader.posy / 1000.0
        vx = (leader.speed * math.cos(leader.theta)) / 30.0
        vy = (leader.speed * math.sin(leader.theta)) / 30.0
        
        # 到goal的相对位置
        dx = (goal.init_x - leader.posx) / 1000.0
        dy = (goal.init_y - leader.posy) / 1000.0
        
        return [x_norm, y_norm, vx, vy, 
                math.cos(leader.theta), math.sin(leader.theta),
                dx, dy]
    
    def _extract_follower_features(self, follower, leader):
        """提取Follower节点特征（8维）"""
        x_norm = follower.posx / 1000.0
        y_norm = follower.posy / 1000.0
        vx = (follower.speed * math.cos(follower.theta)) / 40.0
        vy = (follower.speed * math.sin(follower.theta)) / 40.0
        
        # 到leader的相对位置（编队误差）
        dx = (leader.posx - follower.posx) / 200.0
        dy = (leader.posy - follower.posy) / 200.0
        
        return [x_norm, y_norm, vx, vy,
                math.cos(follower.theta), math.sin(follower.theta),
                dx, dy]
    
    def _build_edges(self, positions, node_types, num_followers):
        """构建图的边"""
        edge_index = []
        edge_attr = []
        edge_types = []
        
        leader_idx = 0
        goal_idx = num_followers + 1
        obstacle_start_idx = num_followers + 2
        
        # 1. Leader → Followers（编队边，总是存在）
        for follower_idx in range(1, num_followers + 1):
            # 双向边
            edge_index.append([leader_idx, follower_idx])
            edge_index.append([follower_idx, leader_idx])
            
            # 边特征：距离、角度等
            edge_feat_lf = self._compute_edge_features(
                positions[leader_idx], positions[follower_idx]
            )
            edge_feat_fl = self._compute_edge_features(
                positions[follower_idx], positions[leader_idx]
            )
            
            edge_attr.append(edge_feat_lf)
            edge_attr.append(edge_feat_fl)
            edge_types.extend([0, 0])  # Formation type
        
        # 2. Leader → Goal（导航边）
        edge_index.append([leader_idx, goal_idx])
        edge_feat = self._compute_edge_features(
            positions[leader_idx], positions[goal_idx]
        )
        edge_attr.append(edge_feat)
        edge_types.append(2)  # Navigation type
        
        # 3. Follower ↔ Follower（协调边，条件性）
        for i in range(1, num_followers + 1):
            for j in range(i+1, num_followers + 1):
                dist = np.linalg.norm(
                    np.array(positions[i]) - np.array(positions[j])
                )
                if dist < self.coordination_radius:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    
                    edge_feat_ij = self._compute_edge_features(positions[i], positions[j])
                    edge_feat_ji = self._compute_edge_features(positions[j], positions[i])
                    
                    edge_attr.append(edge_feat_ij)
                    edge_attr.append(edge_feat_ji)
                    edge_types.extend([1, 1])  # Coordination type
        
        # 4. Agent → Obstacle（避障边，条件性）
        num_obstacles = len(positions) - obstacle_start_idx
        for agent_idx in range(num_followers + 1):
            for obs_offset in range(num_obstacles):
                obs_idx = obstacle_start_idx + obs_offset
                dist = np.linalg.norm(
                    np.array(positions[agent_idx]) - np.array(positions[obs_idx])
                )
                if dist < 100:  # 只在附近时添加边
                    edge_index.append([agent_idx, obs_idx])
                    edge_feat = self._compute_edge_features(
                        positions[agent_idx], positions[obs_idx]
                    )
                    edge_attr.append(edge_feat)
                    edge_types.append(3)  # Avoidance type
        
        return edge_index, edge_attr, edge_types
    
    def _compute_edge_features(self, pos_i, pos_j):
        """计算边特征（4维）"""
        dx = pos_j[0] - pos_i[0]
        dy = pos_j[1] - pos_i[1]
        distance = math.hypot(dx, dy) / 1000.0  # 归一化
        angle = math.atan2(dy, dx) / (2 * math.pi)  # 归一化到[-0.5, 0.5]
        
        return [distance, angle, dx/1000.0, dy/1000.0]
```

**验收标准**：
- ✅ 可以从环境状态成功构建图
- ✅ 图结构正确（节点数、边数）
- ✅ 特征维度正确（节点8维，边4维）
- ✅ 单元测试100%通过

**单元测试**：
```python
def test_graph_builder():
    env = RlGame(n_leader=1, n_follower=4)
    obs, _ = env.reset()
    
    builder = FormationGraphBuilder()
    graph = builder.build_graph(env)
    
    # 验证
    assert graph.num_nodes == 6  # 1 Leader + 4 Followers + 1 Goal
    assert graph.x.shape == (6, 8)
    assert graph.edge_index.shape[1] >= 8  # 至少有编队边
    print("✅ Graph builder测试通过")
```

---

#### **Task 1.3: 异构GAT Actor实现**

**时间**: 2-3天

**实施步骤**：

**文件**: `algorithm/masac/gnn_actor.py`（新建，约350行）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class HeterogeneousGAT_Actor(nn.Module):
    """
    异构图注意力Actor网络
    
    架构：
    1. 异构节点编码器
    2. 2层GAT消息传递
    3. 分角色策略头
    """
    
    def __init__(self, node_dim=8, hidden_dim=64, action_dim=2, 
                 num_heads=4, dropout=0.1):
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        # === 节点类型编码器 ===
        self.leader_encoder = nn.Linear(node_dim, hidden_dim)
        self.follower_encoder = nn.Linear(node_dim, hidden_dim)
        self.goal_encoder = nn.Linear(node_dim, hidden_dim)
        self.obstacle_encoder = nn.Linear(node_dim, hidden_dim)
        
        # === GAT层（消息传递）===
        # Layer 1: 多头注意力
        self.gat1 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True,
            edge_dim=4,  # 边特征维度
            add_self_loops=True
        )
        
        # Layer Norm
        self.ln1 = nn.LayerNorm(hidden_dim * num_heads)
        
        # Layer 2: 单头注意力
        self.gat2 = GATConv(
            in_channels=hidden_dim * num_heads,
            out_channels=hidden_dim,
            heads=1,
            dropout=dropout,
            concat=False,
            edge_dim=4,
            add_self_loops=True
        )
        
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # === Leader策略头 ===
        self.leader_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.leader_mean = nn.Linear(hidden_dim // 2, action_dim)
        self.leader_log_std = nn.Linear(hidden_dim // 2, action_dim)
        
        # === Follower策略头 ===
        self.follower_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.follower_mean = nn.Linear(hidden_dim // 2, action_dim)
        self.follower_log_std = nn.Linear(hidden_dim // 2, action_dim)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """He初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, graph_batch):
        """
        前向传播
        
        Args:
            graph_batch: PyG Batch对象
        
        Returns:
            mean, log_std: [num_agents, action_dim]
        """
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        edge_attr = graph_batch.edge_attr
        node_types = graph_batch.node_type
        
        # === 异构节点编码 ===
        h = torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
        
        leader_mask = (node_types == 0)
        follower_mask = (node_types == 1)
        goal_mask = (node_types == 2)
        obstacle_mask = (node_types == 3)
        
        if leader_mask.any():
            h[leader_mask] = self.leader_encoder(x[leader_mask])
        if follower_mask.any():
            h[follower_mask] = self.follower_encoder(x[follower_mask])
        if goal_mask.any():
            h[goal_mask] = self.goal_encoder(x[goal_mask])
        if obstacle_mask.any():
            h[obstacle_mask] = self.obstacle_encoder(x[obstacle_mask])
        
        # === GAT消息传递 ===
        # Layer 1
        h1 = self.gat1(h, edge_index, edge_attr=edge_attr)
        h1 = self.ln1(h1)
        h1 = F.elu(h1)
        
        # Layer 2 + 残差连接
        h2 = self.gat2(h1, edge_index, edge_attr=edge_attr)
        h2 = self.ln2(h2)
        # h2: [num_nodes, hidden_dim]
        
        # === 策略输出（仅对agent节点）===
        agent_mask = (node_types <= 1)  # Leader和Follower
        agent_embeddings = h2[agent_mask]
        
        # 分离Leader和Follower
        num_leaders = leader_mask.sum().item()
        leader_emb = agent_embeddings[:num_leaders]
        follower_emb = agent_embeddings[num_leaders:]
        
        # Leader策略
        leader_h = self.leader_policy(leader_emb)
        leader_mean = torch.tanh(self.leader_mean(leader_h))  # [-1, 1]
        leader_log_std = self.leader_log_std(leader_h)
        leader_log_std = torch.clamp(leader_log_std, -20, 2)
        
        # Follower策略
        follower_h = self.follower_policy(follower_emb)
        follower_mean = torch.tanh(self.follower_mean(follower_h))
        follower_log_std = self.follower_log_std(follower_h)
        follower_log_std = torch.clamp(follower_log_std, -20, 2)
        
        # 合并
        mean = torch.cat([leader_mean, follower_mean], dim=0)
        log_std = torch.cat([leader_log_std, follower_log_std], dim=0)
        
        return mean, log_std
    
    def get_attention_weights(self, graph_batch):
        """
        获取注意力权重（用于可视化和调试）
        
        Returns:
            attention_weights: GAT层的attention系数
        """
        # 在forward中保存attention_weights
        # 这需要修改GATConv，使用return_attention_weights=True
        pass
```

**验收标准**：
- ✅ 网络可以成功前向传播
- ✅ 输出形状正确：`[num_agents, action_dim]`
- ✅ 输出值域正确：mean∈[-1,1], std>0
- ✅ 梯度可以反向传播
- ✅ 无NaN或Inf

**单元测试**：
```python
def test_het_gat_actor():
    # 构造测试图
    graph = create_test_graph(num_followers=4)
    
    # 创建网络
    actor = HeterogeneousGAT_Actor()
    
    # 前向传播
    mean, log_std = actor(graph)
    
    # 验证
    assert mean.shape == (5, 2)  # 5 agents, 2 actions
    assert (mean >= -1).all() and (mean <= 1).all()
    assert (log_std >= -20).all() and (log_std <= 2).all()
    
    # 反向传播测试
    loss = mean.sum()
    loss.backward()
    
    print("✅ Het-GAT Actor测试通过")
```

---

#### **Task 1.4: 异构GAT Critic实现**

**时间**: 2天

**文件**: `algorithm/masac/gnn_critic.py`（新建，约300行）

```python
class HeterogeneousGAT_Critic(nn.Module):
    """
    异构图注意力Critic网络
    
    CTDE架构：
    - 训练时：使用全局图信息
    - 执行时：仅用局部观测（通过GNN获取）
    """
    
    def __init__(self, node_dim=8, action_dim=2, hidden_dim=64, num_heads=4):
        super().__init__()
        
        # === 共享GNN编码器（可与Actor共享）===
        self.gnn_encoder = GNN_Encoder(node_dim, hidden_dim, num_heads)
        
        # === 动作编码器 ===
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        # === Q网络（双Q架构）===
        # Q1
        self.q1_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Q2
        self.q2_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, graph_batch, actions):
        """
        前向传播
        
        Args:
            graph_batch: PyG图数据
            actions: [num_agents, action_dim]
        
        Returns:
            q1, q2: [num_agents, 1]
        """
        # GNN编码（全局信息）
        node_embeddings = self.gnn_encoder(graph_batch)
        
        # 提取agent节点的嵌入
        agent_mask = (graph_batch.node_type <= 1)
        agent_embeddings = node_embeddings[agent_mask]
        
        # 动作编码
        action_embeddings = self.action_encoder(actions)
        
        # 拼接状态和动作
        state_action = torch.cat([agent_embeddings, action_embeddings], dim=-1)
        
        # 双Q估计
        q1 = self.q1_net(state_action)
        q2 = self.q2_net(state_action)
        
        return q1, q2
```

**验收标准**：
- ✅ Q值估计合理（初期：-1000~0，训练后：-100~100）
- ✅ 双Q网络独立
- ✅ 梯度正常
- ✅ 无数值问题

---

#### **Task 1.5: 训练循环适配**

**时间**: 2-3天

**修改文件**: `algorithm/masac/trainer.py`（修改约200行）

**核心修改**：

```python
class GNN_Trainer(Trainer):
    """GNN版本的Trainer"""
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        
        # 替换Actor和Critic
        self.actor = HeterogeneousGAT_Actor(...)
        self.critic = HeterogeneousGAT_Critic(...)
        self.critic_target = HeterogeneousGAT_Critic(...)
        
        # 图构建器
        self.graph_builder = FormationGraphBuilder()
    
    def choose_action(self, env_state):
        """选择动作（修改为图输入）"""
        # 构建图
        graph = self.graph_builder.build_graph(env_state)
        graph = graph.to(self.device)
        
        # 前向传播
        with torch.no_grad():
            mean, log_std = self.actor(graph)
            std = log_std.exp()
            
            # 重参数化采样
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
            action = torch.tanh(action)  # [-1, 1]
        
        return action.cpu().numpy()
    
    def update_networks(self, batch):
        """更新网络（修改为图batch）"""
        # batch现在是图的列表
        graphs = [self.graph_builder.build_graph(state) for state in batch['states']]
        graph_batch = Batch.from_data_list(graphs).to(self.device)
        
        actions = torch.tensor(batch['actions']).to(self.device)
        rewards = torch.tensor(batch['rewards']).to(self.device)
        
        # ... SAC更新逻辑（与FC版本类似）
```

**验收标准**：
- ✅ 训练循环可以正常运行
- ✅ Loss正常下降
- ✅ 无内存泄漏
- ✅ 与FC版本loss曲线可对比

---

#### **Task 1.6: 经验回放适配**

**时间**: 1天

**修改文件**: `algorithm/masac/buffer.py`（修改约100行）

```python
class GraphReplayBuffer:
    """支持图数据的经验回放缓冲区"""
    
    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, graph, action, reward, next_graph, done):
        """存储图transition"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        # 存储为CPU tensor（节省GPU内存）
        transition = {
            'graph': graph.cpu(),
            'action': action,
            'reward': reward,
            'next_graph': next_graph.cpu(),
            'done': done
        }
        
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """采样batch"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        graphs = []
        actions = []
        rewards = []
        next_graphs = []
        dones = []
        
        for idx in indices:
            transition = self.buffer[idx]
            graphs.append(transition['graph'])
            actions.append(transition['action'])
            rewards.append(transition['reward'])
            next_graphs.append(transition['next_graph'])
            dones.append(transition['done'])
        
        # Batch图数据
        from torch_geometric.data import Batch
        graph_batch = Batch.from_data_list(graphs)
        next_graph_batch = Batch.from_data_list(next_graphs)
        
        return {
            'graphs': graph_batch,
            'actions': torch.tensor(actions),
            'rewards': torch.tensor(rewards),
            'next_graphs': next_graph_batch,
            'dones': torch.tensor(dones)
        }
```

**验收标准**：
- ✅ 可以存储和采样图数据
- ✅ Batch构建正确
- ✅ 内存占用合理（<2GB for 20K transitions）

---

### 2.2 阶段1验收标准（总体）

#### **功能性验收**

| 项目 | 标准 | 验证方法 |
|------|------|---------|
| 图构建 | 正确无误 | 单元测试 |
| 网络前向 | 无错误 | 前向测试 |
| 网络反向 | 梯度正常 | 反向测试 |
| 训练循环 | 可运行 | 50 episodes |

#### **性能验收**（50 episodes快速验证）

| 指标 | 最低标准 | 目标标准 | 验证配置 |
|------|:-------:|:-------:|:-------:|
| **可训练性** | 无崩溃 | Loss下降 | 4F |
| **初步效果** | TIMEOUT<25% | TIMEOUT<15% | 4F |
| **编队趋势** | 有改善迹象 | 编队率>75% | 4F |

#### **完整验收**（200 episodes完整验证）

| 指标 | 最低标准 | 目标标准 | 对比基准 |
|------|:-------:|:-------:|:-------:|
| **编队率** | ≥80% | **≥85%** | v0.4: 70-80% |
| **完成率** | ≥80% | **≥85%** | v0.4: 80% |
| **TIMEOUT率** | ≤10% | **≤7%** | v0.4: <10% |
| **训练稳定性** | 无崩溃 | Loss平滑 | - |

**通过标准**: 4个指标中至少3个达到目标标准

#### **回滚条件**

```
满足以下任一条件立即回滚：
1. TIMEOUT率 > 20%（200 episodes后）
2. 训练崩溃（NaN/Inf）且无法修复
3. 编队率 < 70%（200 episodes后）
4. 实施时间超过3周

回滚目标：v0.4（当前稳定版本）
```

---

## 3. 阶段2：GNN-Transformer混合

### 3.1 实施内容清单

#### **Task 2.1: Transformer编码器实现**

**时间**: 2-3天

**文件**: `algorithm/masac/transformer_layer.py`（新建，约250行）

```python
class SpatialTransformerEncoder(nn.Module):
    """
    空间Transformer编码器
    用于聚合全局上下文
    """
    
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        
        # === 空间位置编码 ===
        self.pos_encoder = SpatialPositionalEncoding(d_model)
        
        # === Transformer编码器层 ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN，更稳定
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
    
    def forward(self, node_embeddings, positions, node_mask=None):
        """
        Args:
            node_embeddings: [batch, num_nodes, d_model] GNN输出
            positions: [batch, num_nodes, 2] 2D坐标
            node_mask: [batch, num_nodes] 可选的mask
        
        Returns:
            output: [batch, num_nodes, d_model]
        """
        # 添加位置编码
        pos_encoding = self.pos_encoder(positions)
        x = node_embeddings + pos_encoding
        
        # Transformer编码
        # 生成attention mask（可选，屏蔽goal和obstacle）
        if node_mask is not None:
            # node_mask: True表示该节点参与attention
            # 转换为Transformer的mask格式
            mask = ~node_mask.unsqueeze(1).expand(-1, node_mask.size(1), -1)
        else:
            mask = None
        
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        return output


class SpatialPositionalEncoding(nn.Module):
    """2D空间位置编码"""
    
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        
        # 可学习的位置编码
        # 将2D坐标映射到d_model维
        self.pos_embedding = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # 或者使用固定的sin/cos编码（Transformer原版）
        # self.register_buffer('pe', self._create_sinusoidal_encoding(d_model))
    
    def forward(self, positions):
        """
        Args:
            positions: [batch, num_nodes, 2] 或 [num_nodes, 2]
        
        Returns:
            encoding: [batch, num_nodes, d_model] 或 [num_nodes, d_model]
        """
        return self.pos_embedding(positions)
    
    def _create_sinusoidal_encoding(self, d_model):
        """创建Sinusoidal位置编码（原版Transformer方法）"""
        # 这是一个高级实现，用于2D空间
        # 需要将(x, y)映射到高维编码
        pass
```

**验收标准**：
- ✅ Transformer可以处理图嵌入序列
- ✅ 位置编码正确添加
- ✅ 输出维度正确
- ✅ Attention可视化正常

---

#### **Task 2.2: GNN-Transformer混合Actor**

**时间**: 3天

**文件**: `algorithm/masac/hybrid_actor.py`（新建，约400行）

```python
class GNN_Transformer_Actor(nn.Module):
    """
    GNN-Transformer混合Actor网络（方案2）
    
    架构：
    1. GNN编码局部编队结构
    2. Transformer聚合全局上下文
    3. 融合层结合两者
    4. 分角色策略头
    """
    
    def __init__(self, node_dim=8, hidden_dim=64, action_dim=2, 
                 num_gat_heads=4, num_trans_heads=4, num_trans_layers=2):
        super().__init__()
        
        # === Stage 1: GNN编码器（局部编队）===
        self.gnn_encoder = HeterogeneousGNN_Encoder(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_heads=num_gat_heads,
            num_layers=2
        )
        
        # === Stage 2: Transformer编码器（全局上下文）===
        self.transformer_encoder = SpatialTransformerEncoder(
            d_model=hidden_dim,
            nhead=num_trans_heads,
            num_layers=num_trans_layers,
            dim_feedforward=hidden_dim * 4
        )
        
        # === Stage 3: 融合层 ===
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # GNN + Trans
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # === Stage 4: 策略头 ===
        self.leader_policy = PolicyHead(hidden_dim, action_dim, 'leader')
        self.follower_policy = PolicyHead(hidden_dim, action_dim, 'follower')
    
    def forward(self, graph_batch):
        # Stage 1: GNN编码
        gnn_embeddings = self.gnn_encoder(graph_batch)  # [num_nodes, hidden_dim]
        
        # Stage 2: Transformer编码
        # 转换为batch格式
        num_nodes = gnn_embeddings.shape[0]
        gnn_seq = gnn_embeddings.unsqueeze(0)  # [1, num_nodes, hidden_dim]
        pos_seq = graph_batch.pos.unsqueeze(0)  # [1, num_nodes, 2]
        
        # 创建mask：只让agent节点参与attention
        agent_mask = (graph_batch.node_type <= 1)  # Leader和Follower
        
        trans_output = self.transformer_encoder(
            gnn_seq, pos_seq, node_mask=agent_mask
        )  # [1, num_nodes, hidden_dim]
        
        trans_embeddings = trans_output.squeeze(0)  # [num_nodes, hidden_dim]
        
        # Stage 3: 融合GNN和Transformer特征
        combined = torch.cat([gnn_embeddings, trans_embeddings], dim=-1)
        fused_embeddings = self.fusion(combined)  # [num_nodes, hidden_dim]
        
        # Stage 4: 策略输出
        agent_embeddings = fused_embeddings[agent_mask]
        
        # 分离Leader和Follower
        num_leaders = (graph_batch.node_type == 0).sum().item()
        leader_emb = agent_embeddings[:num_leaders]
        follower_emb = agent_embeddings[num_leaders:]
        
        # 生成动作
        leader_mean, leader_log_std = self.leader_policy(leader_emb)
        follower_mean, follower_log_std = self.follower_policy(follower_emb)
        
        # 合并
        mean = torch.cat([leader_mean, follower_mean], dim=0)
        log_std = torch.cat([leader_log_std, follower_log_std], dim=0)
        
        return mean, log_std


class PolicyHead(nn.Module):
    """策略头（分角色）"""
    
    def __init__(self, hidden_dim, action_dim, role='leader'):
        super().__init__()
        self.role = role
        
        # 根据角色调整网络结构
        if role == 'leader':
            # Leader：更深的网络，更复杂的决策
            self.policy = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
            )
        else:  # follower
            # Follower：较浅的网络，更快的反应
            self.policy = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim // 2)
            )
        
        self.mean_layer = nn.Linear(hidden_dim // 2, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim // 2, action_dim)
    
    def forward(self, x):
        h = self.policy(x)
        mean = torch.tanh(self.mean_layer(h))
        log_std = torch.clamp(self.log_std_layer(h), -20, 2)
        return mean, log_std
```

**验收标准**：
- ✅ GNN和Transformer可以联合训练
- ✅ 融合层输出合理
- ✅ 端到端梯度传播正常
- ✅ 性能优于纯GNN（阶段1）

---

### 3.2 阶段2验收标准

#### **性能验收**（500 episodes完整训练）

| 指标 | 最低要求 | 目标标准 | 对比基准 |
|------|:-------:|:-------:|:-------:|
| **编队率** | ≥90% | **≥95%** | 阶段1: 85% |
| **完成率** | ≥92% | **≥95%** | 阶段1: 85% |
| **TIMEOUT率** | ≤5% | **≤3%** | 阶段1: 7% |
| **平均步数** | ≤50 | **≤40** | 阶段1: 50-60 |
| **训练稳定性** | Loss收敛 | 平滑收敛 | - |

#### **质量验收**

| 项目 | 标准 |
|------|------|
| **代码质量** | 通过Linter，无警告 |
| **单元测试** | 覆盖率>80% |
| **文档完整性** | API文档+使用示例 |
| **可视化** | Attention权重可视化 |

#### **通过条件**

```
必须同时满足：
1. 编队率 ≥ 95%
2. 完成率 ≥ 95%
3. TIMEOUT率 ≤ 3%
4. 性能稳定（3次实验标准差<5%）

→ 进入阶段3
```

---

## 4. 阶段3：性能优化与扩展

### 4.1 实施内容（可选）

#### **Task 3.1: 多障碍物支持**

**时间**: 1周

**实施内容**：
```python
# 扩展图构建，支持N个障碍物
class MultiObstacleGraphBuilder(FormationGraphBuilder):
    def build_edges(self, ...):
        # 动态添加避障边
        for agent in agents:
            nearby_obstacles = find_obstacles_in_range(agent, radius=100)
            for obs in nearby_obstacles:
                add_avoidance_edge(agent, obs)
```

**验收标准**：
- ✅ 支持10+障碍物
- ✅ 避障成功率>95%
- ✅ 性能不显著下降

#### **Task 3.2: 动态环境适应**

**时间**: 1周

**实施内容**：
```python
# 动态目标、动态障碍物
class DynamicEnvironment(RlGame):
    def step(self, action):
        # 目标位置随时间移动
        self.goal.update_position(t)
        
        # 障碍物移动
        for obs in self.obstacles:
            obs.update_position(t)
        
        # 图需要每步重建
        graph = self.graph_builder.build_graph(...)
```

**验收标准**：
- ✅ 动态环境下编队率>90%
- ✅ 完成率>85%

#### **Task 3.3: 模型压缩与加速**

**时间**: 1周

**实施内容**：
- 知识蒸馏（大模型→小模型）
- 模型剪枝
- 量化（FP32→FP16）

**验收标准**：
- ✅ 推理速度>30 FPS
- ✅ 模型大小<5MB
- ✅ 性能损失<3%

---

## 5. 验收标准体系

### 5.1 三级验收体系

```
Level 1: 单元测试（开发中）
├─ 每个组件独立测试
├─ 覆盖率>80%
└─ CI自动运行

Level 2: 集成测试（阶段末）
├─ 端到端训练测试
├─ 50-200 episodes验证
└─ 性能基准对比

Level 3: 系统测试（最终）
├─ 完整500 episodes训练
├─ 多配置测试（1F-6F）
└─ A/B对比测试
```

### 5.2 性能基准矩阵

#### **阶段1验收基准**

| 配置 | 编队率 | 完成率 | TIMEOUT率 | 平均步数 |
|------|:-----:|:-----:|:--------:|:-------:|
| **1F** | ≥95% | ≥95% | ≤2% | ≤35 |
| **2F** | ≥90% | ≥90% | ≤5% | ≤40 |
| **3F** | ≥88% | ≥88% | ≤6% | ≤45 |
| **4F** | ≥85% | ≥85% | ≤7% | ≤50 |

#### **阶段2验收基准**

| 配置 | 编队率 | 完成率 | TIMEOUT率 | 平均步数 |
|------|:-----:|:-----:|:--------:|:-------:|
| **1F** | ≥98% | ≥98% | ≤1% | ≤30 |
| **2F** | ≥97% | ≥97% | ≤2% | ≤35 |
| **3F** | ≥96% | ≥96% | ≤2% | ≤38 |
| **4F** | ≥95% | ≥95% | ≤3% | ≤40 |
| **5F** | ≥93% | ≥93% | ≤4% | ≤45 |
| **6F** | ≥90% | ≥90% | ≤5% | ≤50 |

### 5.3 定性验收标准

#### **可解释性验收**

```python
# 1. Attention权重可视化
visualize_attention_weights(graph, attention_matrix)
# 要求：
# - Leader对Follower的attention > 0.6
# - Follower对Leader的attention > 0.7
# - 掉队的Follower attention更高

# 2. 编队质量分析
formation_quality = compute_formation_metrics(trajectories)
# 要求：
# - 编队形状稳定（方差<10）
# - 编队中心接近Leader
# - 无Follower长期掉队（>10步）

# 3. 决策可理解性
explain_decision(state, action, attention_weights)
# 要求：
# - 接近goal时，Leader加速
# - 有follower掉队时，Leader减速
# - 障碍物接近时，相关agent注意力↑
```

#### **鲁棒性验收**

| 测试场景 | 通过标准 |
|---------|---------|
| 随机初始化(100次) | 成功率>90% |
| 不同follower数量(1-6) | 所有配置达标 |
| 不同障碍物位置(10种) | 避障率>95% |
| 极端距离(goal很远) | 可收敛 |

---

## 6. 风险管理与回滚

### 6.1 风险监控指标

#### **训练过程监控**

```python
# 实时监控的危险信号
DANGER_SIGNALS = {
    'nan_loss': "Loss出现NaN",
    'exploding_grad': "梯度爆炸(norm>10)",
    'timeout_spike': "TIMEOUT率突然>30%",
    'reward_collapse': "奖励崩溃（全<-1000）",
    'no_convergence': "200 ep后无收敛迹象"
}

# 监控频率
check_every_n_episodes = 10

# 触发回滚的阈值
ROLLBACK_THRESHOLDS = {
    'timeout_rate_50ep': 0.25,   # 前50ep TIMEOUT>25%
    'timeout_rate_200ep': 0.15,  # 200ep后仍>15%
    'formation_rate': 0.60,      # 编队率<60%
    'crash_count': 3             # 崩溃3次
}
```

### 6.2 分阶段回滚策略

```
┌─────────────────────────────────────────┐
│  阶段1回滚策略                          │
├─────────────────────────────────────────┤
│                                         │
│  触发条件：                             │
│  - 50ep后TIMEOUT>25%                    │
│  - 训练崩溃>3次                         │
│  - 性能明显劣于v0.4                     │
│                                         │
│  回滚目标：v0.4（11维FC）               │
│                                         │
│  回滚步骤：                             │
│  1. 停止GNN训练                         │
│  2. 切换回FC模型                        │
│  3. 分析失败原因                        │
│  4. 调整方案或放弃GNN                   │
│                                         │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  阶段2回滚策略                          │
├─────────────────────────────────────────┤
│                                         │
│  触发条件：                             │
│  - 性能不如阶段1                        │
│  - 训练时间>3倍阶段1                    │
│  - 无法稳定训练                         │
│                                         │
│  回滚目标：阶段1（Het-GAT）             │
│                                         │
│  备选：降级Transformer                  │
│  - 减少层数（2→1）                      │
│  - 减少头数（4→2）                      │
│  - 简化融合层                           │
│                                         │
└─────────────────────────────────────────┘
```

### 6.3 A/B测试框架

```python
class ABTestFramework:
    """A/B对比测试框架"""
    
    def run_ab_test(self, model_a, model_b, num_episodes=100):
        """
        运行A/B测试
        
        Args:
            model_a: 对照组（如FC）
            model_b: 实验组（如GNN）
            num_episodes: 测试episodes数
        
        Returns:
            comparison_report: 详细对比报告
        """
        results_a = self.evaluate(model_a, num_episodes)
        results_b = self.evaluate(model_b, num_episodes)
        
        # 统计检验
        timeout_pvalue = ttest_ind(results_a.timeout_rates, results_b.timeout_rates)
        formation_pvalue = ttest_ind(results_a.formation_rates, results_b.formation_rates)
        
        # 判定显著性
        if timeout_pvalue < 0.05 and results_b.timeout_rate < results_a.timeout_rate:
            print("✅ 模型B显著优于模型A（TIMEOUT率）")
        
        if formation_pvalue < 0.05 and results_b.formation_rate > results_a.formation_rate:
            print("✅ 模型B显著优于模型A（编队率）")
        
        return comparison_report
```

---

## 7. 资源需求评估

### 7.1 人力资源

| 阶段 | 需求 | 角色 |
|------|------|------|
| **阶段1** | 1人 × 2周 | 熟悉PyTorch和RL |
| **阶段2** | 1-2人 × 3周 | 熟悉GNN和Transformer |
| **阶段3** | 1人 × 2周 | 优化专家（可选）|

### 7.2 计算资源

#### **训练资源**

| 阶段 | GPU需求 | 内存需求 | 训练时间 |
|------|:------:|:-------:|:-------:|
| **阶段1** | RTX 2060+ (6GB) | 8GB RAM | 1-2小时 |
| **阶段2** | RTX 3060+ (8GB) | 16GB RAM | 2-3小时 |
| **阶段3** | RTX 3080+ (10GB) | 16GB RAM | 3-5小时 |

#### **开发资源**

```
软件环境：
- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.5+
- CUDA 11.8+ (GPU训练)

硬件最低配置：
- CPU: 4核+
- RAM: 16GB
- GPU: 6GB+ VRAM
- 硬盘: 20GB+
```

### 7.3 时间成本预估

```
阶段1：
├─ 开发时间：1.5周
├─ 调试时间：0.5周
├─ 训练验证：50+200 episodes = 3小时
└─ 总计：2周

阶段2：
├─ 开发时间：2.5周
├─ 调试时间：1周
├─ 训练验证：500 episodes × 2 = 6小时
└─ 总计：3.5周

阶段3（可选）：
├─ 研究时间：1周
├─ 开发时间：1周
├─ 实验验证：1周
└─ 总计：3周

总时间（含阶段3）：8.5周
总时间（仅阶段1-2）：5.5周
```

---

## 8. 详细实施检查清单

### 8.1 阶段1详细清单

#### **Week 1**

**Day 1-2: 环境准备**
- [ ] 安装PyTorch Geometric
- [ ] 验证GPU支持
- [ ] 创建项目分支`feature/gnn-architecture`
- [ ] 更新requirements.txt

**Day 3-4: 图数据结构**
- [ ] 实现`graph_builder.py`
- [ ] 编写节点特征提取函数
- [ ] 编写边构建函数
- [ ] 单元测试（10个测试用例）

**Day 5: 集成测试**
- [ ] 从环境成功构建图
- [ ] 可视化图结构
- [ ] 性能测试（构图时间<1ms）

#### **Week 2**

**Day 1-3: GNN Actor实现**
- [ ] 实现异构节点编码器
- [ ] 实现GAT消息传递层
- [ ] 实现分角色策略头
- [ ] 单元测试

**Day 4-5: GNN Critic实现**
- [ ] 实现Critic网络
- [ ] 双Q架构
- [ ] 单元测试

**Day 6-7: 训练适配**
- [ ] 修改Trainer支持图输入
- [ ] 修改Buffer存储图数据
- [ ] 端到端测试（50 episodes）

### 8.2 阶段2详细清单

#### **Week 3**

**Day 1-3: Transformer实现**
- [ ] 空间位置编码
- [ ] Transformer编码器层
- [ ] 单元测试

**Day 4-5: 混合架构**
- [ ] GNN-Trans融合层
- [ ] 混合Actor实现
- [ ] 前向测试

#### **Week 4-5**

**Day 1-5: 集成与调试**
- [ ] 混合Critic实现
- [ ] 端到端集成
- [ ] 训练调试（50 episodes）
- [ ] 超参数初步调优

**Day 6-10: 完整训练**
- [ ] 4F配置完整训练（500 ep）
- [ ] 3F配置验证（500 ep）
- [ ] 性能对比分析
- [ ] 编写验收报告

---

## 9. 超参数配置建议

### 9.1 阶段1超参数

```yaml
# configs/masac/gnn_config_stage1.yaml

model:
  architecture: 'het-gat'
  
  # 图结构
  formation_radius: 50
  coordination_radius: 100
  
  # GNN参数
  node_dim: 8
  hidden_dim: 64
  num_gat_layers: 2
  num_gat_heads: 4
  gat_dropout: 0.1
  
  # 策略头
  leader_policy_dim: 64
  follower_policy_dim: 64

training:
  # 学习率（可能需要降低）
  policy_lr: 5.0e-4      # FC: 1e-3 → GNN: 5e-4
  q_lr: 2.0e-4           # FC: 3e-4 → GNN: 2e-4
  value_lr: 2.0e-3       # FC: 3e-3 → GNN: 2e-3
  
  # Warmup
  lr_warmup_episodes: 50
  
  # 其他
  batch_size: 64         # FC: 128 → GNN: 64（图batch更大）
  memory_capacity: 20000
  gamma: 0.9
  tau: 1.0e-2
```

### 9.2 阶段2超参数

```yaml
# configs/masac/gnn_trans_config_stage2.yaml

model:
  architecture: 'gnn-transformer'
  
  # GNN参数（继承阶段1）
  gnn_hidden_dim: 64
  num_gat_layers: 2
  num_gat_heads: 4
  
  # Transformer参数
  trans_d_model: 64
  trans_nhead: 4
  trans_num_layers: 2
  trans_dim_feedforward: 256
  trans_dropout: 0.1
  
  # 融合层
  fusion_method: 'concat'  # 'concat', 'add', 'gate'

training:
  # 学习率（进一步降低）
  policy_lr: 3.0e-4
  q_lr: 1.5e-4
  value_lr: 1.5e-3
  
  # Warmup更长
  lr_warmup_episodes: 100
  
  # 正则化
  weight_decay: 1.0e-5
  grad_clip_norm: 1.0
```

---

## 10. 测试与验证计划

### 10.1 单元测试清单

**图构建测试**：
```python
tests/test_graph_builder.py:
- test_leader_node_features()
- test_follower_node_features()
- test_edge_construction()
- test_dynamic_edge_addition()
- test_batch_construction()
```

**网络测试**：
```python
tests/test_gnn_networks.py:
- test_gat_forward()
- test_gat_backward()
- test_heterogeneous_encoding()
- test_policy_output_range()
- test_critic_q_value_range()
```

**集成测试**：
```python
tests/test_gnn_integration.py:
- test_env_to_graph_pipeline()
- test_action_selection()
- test_training_step()
- test_model_save_load()
```

### 10.2 性能测试协议

#### **快速验证（50 episodes）**

```bash
# 测试脚本
python scripts/gnn/quick_test.py \
    --config configs/masac/gnn_config_stage1.yaml \
    --n_follower 4 \
    --ep_max 50 \
    --seed 42

# 验收标准（宽松，仅判断可行性）
期望结果：
- 可完整运行50 episodes
- TIMEOUT率 < 30%
- 有改善趋势（loss下降）
- 无崩溃
```

#### **完整验证（200 episodes）**

```bash
# 正式训练
python scripts/gnn/train_gnn.py \
    --config configs/masac/gnn_config_stage1.yaml \
    --n_follower 4 \
    --ep_max 200 \
    --seed 42

# 验收标准（严格）
期望结果：
- TIMEOUT率 ≤ 7%
- 编队率 ≥ 85%
- 完成率 ≥ 85%
```

#### **多配置测试**

```bash
# 测试1F-4F所有配置
for n_f in {1,2,3,4}; do
    python scripts/gnn/train_gnn.py \
        --n_follower $n_f \
        --ep_max 200 \
        --seed 42
done

# 生成对比报告
python scripts/gnn/generate_comparison_report.py
```

### 10.3 对比基准测试

```python
# 对比测试框架
def run_comparison_test():
    """
    FC vs GNN完整对比
    """
    configs = [
        ('FC_v0.4', 'configs/masac/default.yaml'),
        ('Het-GAT', 'configs/masac/gnn_config_stage1.yaml'),
        ('GNN-Trans', 'configs/masac/gnn_trans_config_stage2.yaml')
    ]
    
    results = {}
    for name, config in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")
        
        # 运行5次取平均（消除随机性）
        metrics = []
        for seed in range(5):
            result = train_and_evaluate(config, seed=seed)
            metrics.append(result)
        
        # 统计
        avg_metrics = aggregate_metrics(metrics)
        results[name] = avg_metrics
    
    # 生成对比报告
    generate_report(results)
    
    # 示例输出：
    # ┌──────────────────────────────────────────┐
    # │  Architecture Comparison                 │
    # ├──────────────────────────────────────────┤
    # │  Metric      │ FC_v0.4 │ Het-GAT │ GNN-Trans │
    # ├──────────────────────────────────────────┤
    # │  Formation%  │  78%    │  87%  ⬆️│  96%  ⬆️ │
    # │  Success%    │  82%    │  86%  ⬆️│  95%  ⬆️ │
    # │  TIMEOUT%    │  8%     │  6%   ⬇️│  2%   ⬇️ │
    # │  Avg Steps   │  45     │  42   ⬇️│  35   ⬇️ │
    # │  Train Time  │  30min  │  40min  │  55min  │
    # └──────────────────────────────────────────┘
```

---

## 11. 文档交付清单

### 11.1 技术文档

| 文档 | 内容 | 负责阶段 |
|------|------|:-------:|
| **API文档** | 所有新类和函数的docstring | 各阶段 |
| **架构设计文档** | 网络结构图、数据流图 | 阶段1 |
| **训练指南** | 如何训练GNN模型 | 阶段1 |
| **性能对比报告** | FC vs GNN详细对比 | 阶段2 |
| **最佳实践文档** | 超参数调优经验 | 阶段2 |

### 11.2 代码交付

```
新增文件：
├── rl_env/
│   └── graph_builder.py              # 图构建器
├── algorithm/masac/
│   ├── gnn_actor.py                  # GNN Actor
│   ├── gnn_critic.py                 # GNN Critic  
│   ├── transformer_layer.py          # Transformer层
│   ├── hybrid_actor.py               # 混合Actor
│   └── hybrid_critic.py              # 混合Critic
├── scripts/gnn/
│   ├── train_gnn.py                  # GNN训练脚本
│   ├── test_gnn.py                   # GNN测试脚本
│   └── visualize_attention.py        # 注意力可视化
├── configs/masac/
│   ├── gnn_config_stage1.yaml        # 阶段1配置
│   └── gnn_trans_config_stage2.yaml  # 阶段2配置
└── tests/
    ├── test_graph_builder.py
    ├── test_gnn_networks.py
    └── test_gnn_integration.py
```

---

## 12. 成功案例与最佳实践

### 12.1 预期成功指标

**阶段1成功标志**：
```
✅ 编队率提升至85-90%（+10-15%）
✅ TIMEOUT率降至5-7%（-30%）
✅ 训练稳定，可复现
✅ 代码质量高，可维护
```

**阶段2成功标志**：
```
✅ 编队率达到95%+（接近理论极限）
✅ TIMEOUT率<3%（行业领先水平）
✅ 支持6+follower（扩展性验证）
✅ 论文级性能，可发表
```

### 12.2 失败案例预案

**场景1：GNN训练不稳定**

```
症状：Loss震荡，NaN频繁出现
原因：学习率过高，梯度爆炸
解决：
1. 降低学习率（/2）
2. 增加梯度裁剪（max_norm=0.5）
3. 使用Pre-LN（LayerNorm前置）
4. 增加warmup期（50→100 ep）
```

**场景2：性能不升反降**

```
症状：GNN性能<FC
原因：网络容量不足或超参数不当
解决：
1. 增加hidden_dim（64→128）
2. 增加GAT层数（2→3）
3. 调整attention头数（4→8）
4. 使用预训练（从FC初始化）
```

**场景3：训练时间过长**

```
症状：单episode>10秒
原因：图batch效率低或网络过大
解决：
1. 优化图构建（缓存、并行）
2. 减少Transformer层（2→1）
3. 使用更小的batch_size
4. GPU优化（混合精度训练）
```

---

## 13. 迁移与回退策略

### 13.1 从FC到GNN的迁移

#### **知识迁移**

```python
def transfer_fc_to_gnn(fc_actor, gnn_actor):
    """
    将FC Actor的知识迁移到GNN Actor
    """
    # FC Actor结构：
    # fc1: [11, 256]
    # fc2: [256, 256]  
    # mean: [256, 2]
    
    # GNN Actor结构：
    # node_encoder: [8, 64]
    # gat1: [64, 64×4]
    # policy: [64, 2]
    
    # 策略1：初始化策略头
    # 将FC的mean层权重映射到GNN的policy头
    with torch.no_grad():
        # FC mean层的前64个神经元
        fc_mean_weights = fc_actor.mean_layer.weight[:, :64]
        
        # 初始化GNN Leader策略头
        gnn_actor.leader_mean.weight.copy_(fc_mean_weights)
        
        # 类似地初始化Follower
        gnn_actor.follower_mean.weight.copy_(fc_mean_weights)
    
    print("✅ FC→GNN知识迁移完成")

# 使用
fc_actor = load_fc_model('runs/exp_baseline_20251031/leader.pth')
gnn_actor = HeterogeneousGAT_Actor()
transfer_fc_to_gnn(fc_actor, gnn_actor)
```

**预期效果**：
- 加速GNN训练（减少50-100 episodes）
- 更稳定的初始性能
- 降低失败风险

### 13.2 版本并存策略

```python
# 支持多架构并存
class MultiArchitectureTrainer:
    def __init__(self, architecture='fc'):
        if architecture == 'fc':
            self.actor = FC_Actor()
            self.critic = FC_Critic()
        elif architecture == 'het-gat':
            self.actor = HeterogeneousGAT_Actor()
            self.critic = HeterogeneousGAT_Critic()
        elif architecture == 'gnn-trans':
            self.actor = GNN_Transformer_Actor()
            self.critic = GNN_Transformer_Critic()
    
    # 统一接口
    def train(self, ...):
        # 自动适配不同架构
        pass
```

---

## 14. 最终验收与部署

### 14.1 最终验收清单

#### **性能指标**

- [ ] 4F编队率 ≥ 95%
- [ ] 4F完成率 ≥ 95%
- [ ] 4F TIMEOUT率 ≤ 3%
- [ ] 3F编队率 ≥ 97%
- [ ] 可扩展到6F，性能下降<10%

#### **质量指标**

- [ ] 代码覆盖率 > 80%
- [ ] 文档完整（API + 使用指南）
- [ ] 无Linter警告
- [ ] Git提交历史清晰

#### **可靠性指标**

- [ ] 5次训练标准差 < 5%
- [ ] 不同种子结果一致
- [ ] 无已知BUG
- [ ] 回归测试全部通过

### 14.2 部署决策

```
最终评审会议（Week 6或Week 9）：

参会者：项目负责人、技术lead、测试工程师

决策流程：
1. 展示性能对比数据
2. 展示训练稳定性
3. 代码审查通过
4. 讨论风险和收益

决策结果：
├─ ✅ 通过 → 部署到生产，废弃FC
├─ ⚠️  条件通过 → 保留FC作为备份
└─ ❌ 不通过 → 继续使用FC，GNN作为研究

部署步骤：
1. 创建release分支
2. 合并feature分支
3. 打tag（如v1.0-gnn）
4. 更新README和文档
5. 归档旧模型
```

---

## 15. 附录

### A. 关键Milestones

| Milestone | Week | 交付物 | 验收人 |
|-----------|:---:|--------|--------|
| M1: 图数据结构完成 | 1 | graph_builder.py + 测试 | Tech Lead |
| M2: GNN Actor/Critic完成 | 2 | gnn_actor.py + gnn_critic.py | Tech Lead |
| M3: 阶段1验收 | 2 | 200ep训练报告 | 项目负责人 |
| M4: Transformer集成 | 4 | hybrid_actor.py | Tech Lead |
| M5: 阶段2验收 | 5 | 500ep性能报告 | 项目负责人 |
| M6: 最终部署 | 6 | 生产版本 | 全体 |

### B. 沟通计划

- **每周例会**：进度汇报、问题讨论
- **每阶段末评审**：是否进入下一阶段
- **技术难题会议**：遇到阻塞时立即召开
- **最终验收会**：部署前的正式评审

### C. 风险登记册

| 风险ID | 描述 | 概率 | 影响 | 缓解措施 | 负责人 |
|--------|------|:---:|:---:|---------|--------|
| R1 | GNN训练不稳定 | 40% | 高 | 学习率调整、warmup | Dev |
| R2 | 性能不达标 | 30% | 高 | A/B测试、回滚 | PM |
| R3 | 时间超期 | 50% | 中 | 砍掉阶段3 | PM |
| R4 | GPU资源不足 | 20% | 低 | 云GPU | Ops |

---

## 16. 成功标准总结

### 16.1 分阶段成功定义

```
阶段1成功 =
    编队率 ≥ 85% AND
    完成率 ≥ 85% AND  
    TIMEOUT率 ≤ 7% AND
    训练稳定（无崩溃）

阶段2成功 =
    编队率 ≥ 95% AND
    完成率 ≥ 95% AND
    TIMEOUT率 ≤ 3% AND
    优于阶段1

项目成功 =
    达到阶段2成功标准 OR
    阶段1成功 + 明确的后续改进路径
```

### 16.2 投资回报评估

| 投入 | 阶段1 | 阶段2 | 总计 |
|------|:----:|:----:|:---:|
| **开发时间** | 2周 | 3.5周 | 5.5周 |
| **GPU时间** | 3小时 | 6小时 | 9小时 |
| **人力成本** | 1人×2周 | 1人×3.5周 | 1人×5.5周 |

| 收益 | 预期值 |
|------|-------|
| **编队率提升** | +15-25% |
| **完成率提升** | +5-15% |
| **可扩展性** | 支持10+follower |
| **论文价值** | 可发表 |

**ROI**: 高（性能提升显著，适合论文发表）

---

## 17. 下一步行动

### 17.1 立即行动（本周）

```
✅ Day 1-2: 
   - 学习PyTorch Geometric基础
   - 阅读GAT论文
   - 环境准备

✅ Day 3-4:
   - 实现graph_builder.py
   - 单元测试
   - 可视化图结构

✅ Day 5-7:
   - 开始实施GNN Actor
   - 前向传播测试
```

### 17.2 决策点

**Week 2 Decision Point**：
- 如果单元测试全部通过 → 继续
- 如果性能初步验证OK → 继续
- 如果遇到重大技术障碍 → 评估是否继续

**Week 5 Decision Point**：
- 如果阶段2成功 → 考虑阶段3或直接部署
- 如果不如预期 → 使用阶段1版本
- 如果完全失败 → 回退FC

---

**文档版本**: v1.0  
**实施路线图状态**: 🟢 Ready to Execute  
**预期成功率**: 70-80% (阶段1), 60-70% (阶段2)  
**建议**: 务必渐进实施，每阶段充分验证


