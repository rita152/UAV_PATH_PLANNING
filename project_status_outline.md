# 无人机编队路径规划项目 - 设计问题分析
> 状态变量与奖励函数设计不合理分析

---

## 第1页：项目现状与核心问题

### 📊 测试数据概览（4F配置）

| 指标 | 数值 | 状态 |
|:-----|:----:|:---:|
| 任务完成率 | **92%** | ✅ 优秀 |
| 编队保持率 | **1%** | 🔴 严重不足 |
| 平均飞行时间 | 57步 | ✅ 高效 |

### 🎯 Agent学到的策略

**Leader行为：**
- "忽略Follower，直奔目标"
- 全速前进，不等待
- 单独完成任务

**Follower行为：**
- 努力追赶Leader
- 速度不足，追不上
- 松散跟随

### ⚡ 核心问题
**任务完成率高但编队率极低 = 环境设计问题**

---

## 第2页：问题1 - 奖励函数设计严重失衡

### 🔴 当前奖励函数结构

| 奖励组件 | 数值 | 占比 | 说明 |
|---------|:----:|:---:|:-----|
| **目标奖励** | +1000 | **99%** | Leader到达目标点 |
| **编队奖励** | +1~4 | **0.4%** | 速度匹配奖励 |
| 距离惩罚 | -0.005×dist | 0.5% | 编队距离惩罚 |
| 时间惩罚 | -0.2/step | 微小 | 步数惩罚 |

**权重比：目标 : 编队 = 250 : 1**

### 📐 数学证明：为什么Leader不等待？

**策略A（等待编队）：**
```
编队奖励：+120，目标奖励：+1000，时间：60步
奖励效率：1108 ÷ 60 = 18.5 reward/step
```

**策略B（直接冲刺）：**
```
目标奖励：+1000，时间：50步
奖励效率：990 ÷ 50 = 19.8 reward/step
```

**结论：策略B > 策略A → SAC理性学到"忽略编队"策略**

### 💥 设计缺陷

1. **编队只是评估指标，不是训练目标**
   - 奖励函数未正确编码"编队"目标
   - Leader缺少等待Follower的动力

2. **编队非任务必要条件**
   - Leader可以单独完成任务
   - 编队成为可选项而非必需品

3. **编队判断过于严格**
   ```python
   if formation_count == n_follower:  # 全员编队才计数
       team_counter += 1
   ```
   - 4个Follower同时编队概率仅6.25%（理论值）
   - 实际更低（1%），因为Leader不等待

---

## 第3页：问题2 - 状态变量设计不合理

### 🧠 状态维度对比（均为7维）

| 角色 | 状态变量 | 关键问题 |
|------|---------|---------|
| **Leader** | `[x, y, speed, angle,`<br>`goal_x, goal_y, obstacle_flag]` | ❌ **无Follower感知** |
| **Follower** | `[x, y, speed, angle,`<br>`leader_x, leader_y, leader_speed]` | ✅ 有Leader感知 |

### 🔴 核心设计缺陷

#### **缺陷1：Leader完全"看不见"Follower**

Leader状态中缺失的关键信息：
- ❌ Follower位置（avg_follower_distance）
- ❌ 编队质量（formation_quality）  
- ❌ 掉队情况（follower_behind）

**后果：** Leader做决策时无法考虑编队，只能基于目标点导航

#### **缺陷2：单向依赖，弱耦合**

```
Leader → 目标（99%权重）
         ↓ 极弱关联（1%）
      Follower → Leader（100%跟随）
```

**问题：**
- Follower依赖Leader（强耦合）
- Leader不依赖Follower（弱耦合）
- 无团队协同奖励机制

#### **缺陷3：缺失团队协同机制**

1. **无全局视角**：Leader只知道自己和目标，不知道团队状态
2. **无协同奖励**：没有"团队成功"的奖励概念
3. **信息不对称**：Follower知道Leader位置，Leader不知道Follower位置

### 💡 本质结论

> **当前系统本质上是单智能体任务，编队只是附加要求**

**问题根源：**
- 奖励函数主导：目标99% vs 编队0.4%
- 状态变量限制：Leader无感知 → 无法协同
- 协同机制缺失：单向依赖 + 无团队奖励

**结果：**
- SAC算法完美学到了"最优策略" = 忽略编队
- 但这个"最优"不符合真实目标（编队协同）

---

## 第4页：改进方向1 - 状态变量与奖励函数重新设计

### 🔧 状态变量改进方案

#### **Leader状态扩展（7维 → 9维）**

| 当前状态 (7维) | 改进状态 (9维) | 新增维度 |
|--------------|--------------|---------|
| `[x, y, speed, angle,`<br>`goal_x, goal_y,`<br>`obstacle_flag]` | `[x, y, speed, angle,`<br>`goal_x, goal_y,`<br>`obstacle_flag,`<br>**`avg_follower_dist,`**<br>**`formation_ratio`**`]` | ✅ 平均Follower距离<br>✅ 编队完成度 |

**改进价值：**
- Leader获得Follower感知能力
- 可基于编队状态做决策
- 实现真正的协同控制

#### **Follower状态保持（7维）**
- 已包含Leader感知信息，无需修改
- 继续使用：`[x, y, speed, angle, leader_x, leader_y, leader_speed]`

### 💰 奖励函数重新平衡

#### **奖励权重调整**

| 组件 | 当前值 | 改进值 | 调整倍数 |
|-----|:-----:|:-----:|:-------:|
| **目标奖励** | +1000 | **+500** | ↓ 50% |
| **编队奖励（单体）** | +1.0 | **+10.0** | ↑ 10x |
| **团队编队奖励（新增）** | 0 | **+50.0** | ✅ 新增 |
| **编队距离系数** | -0.005 | **-0.05** | ↑ 10x |

**新权重比：** `目标 : 编队 = 500 : (10×4 + 50) = 500 : 90 ≈ 5.5 : 1`

#### **新增团队协同奖励机制**

```python
# 渐进式编队奖励
formation_ratio = formation_count / n_follower
team_bonus = 50.0 * formation_ratio  # 按比例给奖励

# Leader奖励
r_leader = goal_reward * 0.5  # 降低目标主导性
         + formation_bonus * 10.0  # 提升编队重要性
         + team_bonus  # 团队协同奖励

# 编队成为高价值行为
```

**预期效果：** 编队率从1%提升至30-50%

---

## 第5页：改进方向2 - GNN-Transformer混合架构

### 🚀 架构升级：从MLP到GNN-Transformer

#### **当前架构问题**
- ❌ **MLP无法建模智能体关系**：Leader-Follower关系被扁平化
- ❌ **状态拼接方式原始**：无法捕捉动态交互
- ❌ **缺乏注意力机制**：无法区分重要/不重要的Follower

#### **GNN-Transformer混合架构设计**

```
输入层（图结构）
    ↓
【GNN模块】图神经网络 - 建模Leader-Follower关系
    ├─ 节点特征：每个Agent的状态向量
    ├─ 边特征：距离、速度差、角度差
    └─ 图卷积：聚合邻居信息（2-3层）
    ↓
【Transformer模块】自注意力机制 - 全局协同
    ├─ Multi-Head Attention：关注关键Follower
    ├─ Position Encoding：编码智能体ID
    └─ Feed Forward：特征提取
    ↓
【策略/价值网络】Actor-Critic输出
    ├─ Actor：动作分布 [angle_change, speed_change]
    └─ Critic：状态价值估计
```

### 🎯 架构优势分析

| 维度 | 当前MLP | GNN-Transformer | 提升 |
|-----|--------|----------------|-----|
| **关系建模** | ❌ 无 | ✅ 显式图结构 | 🔥🔥🔥 |
| **注意力机制** | ❌ 无 | ✅ 多头注意力 | 🔥🔥🔥 |
| **可扩展性** | ❌ 差（固定维度） | ✅ 强（动态节点） | 🔥🔥 |
| **全局视角** | ❌ 局部 | ✅ 全局 | 🔥🔥 |
| **编队感知** | ❌ 弱 | ✅ 强 | 🔥🔥🔥 |

### 🔬 核心技术模块

#### **1. GNN图构建**
```python
# 构建异构图
Graph = {
    'nodes': [Leader, F1, F2, F3, F4],
    'edges': [
        (Leader → F1), (Leader → F2), ...,  # Leader到Follower
        (F1 → Leader), (F2 → Leader), ...,  # Follower到Leader
        (F1 → F2), (F2 → F3), ...           # Follower之间
    ],
    'node_features': [x, y, speed, angle, ...],
    'edge_features': [distance, speed_diff, angle_diff]
}
```

#### **2. Graph Attention Network (GAT)**
```python
# 图注意力层
h_i' = σ(Σ α_ij W h_j)
      j∈N(i)

# α_ij: 节点j对节点i的注意力权重
# Leader自动学习关注哪些Follower更重要
```

#### **3. Transformer全局建模**
```python
# 多头注意力
Attention(Q, K, V) = softmax(QK^T / √d_k) V

# Leader通过Attention获得全局编队状态
# 每个Follower通过Attention感知其他Follower
```
---

**END**

