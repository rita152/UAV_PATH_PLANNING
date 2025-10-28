# UAV路径规划系统 (UAV Path Planning)

基于深度强化学习（SAC算法）的多智能体无人机协同路径规划系统

## 📖 项目简介

本项目实现了一个基于Soft Actor-Critic (SAC)算法的无人机路径规划系统，支持多智能体协同、动态避障和目标追踪。系统使用PyTorch构建深度神经网络，通过强化学习训练智能体在复杂环境中完成路径规划任务。

采用 **Leader-Follower 协同控制架构**，Leader 负责路径规划和目标导航，Follower 负责跟随 Leader 并保持编队。

### 主要特性

- ✅ **Leader-Follower 协同**：支持 Leader-Follower 编队控制
- ✅ **动态避障**：实时检测并规避障碍物
- ✅ **目标导航**：引导 Leader 到达指定目标点
- ✅ **编队保持**：Follower 自动跟随 Leader 保持队形
- ✅ **可视化仿真**：基于Pygame的实时可视化
- ✅ **自动训练**：支持离线训练和在线测试
- ✅ **模型保存**：自动保存训练好的神经网络模型

## 🛠️ 技术栈

- **深度学习框架**：PyTorch
- **强化学习环境**：Gymnasium (OpenAI Gym 的维护版本)
- **可视化**：Pygame, Matplotlib
- **数值计算**：NumPy
- **编程语言**：Python 3.7+

## 📦 安装说明

### 环境要求

- Python 3.7 或更高版本
- Anaconda 或 Miniconda
- CUDA（可选，用于GPU加速）

### 创建 Conda 环境

**重要**：本项目使用专用的 conda 环境 `UAV_PATH_PLANNING`

```bash
# 克隆项目
git clone https://github.com/your-username/UAV_PATH_PLANNING.git
cd UAV_PATH_PLANNING

# 创建 conda 环境
conda create -n UAV_PATH_PLANNING python=3.8

# 激活环境
conda activate UAV_PATH_PLANNING

# 安装依赖包（方式1：使用 requirements.txt，推荐）
pip install -r requirements.txt

# 或者手动安装（方式2）
pip install torch torchvision
pip install gymnasium pygame numpy matplotlib
```

### ⚠️ 重要提示

**每次运行代码前，必须先激活 conda 环境**：
```bash
conda activate UAV_PATH_PLANNING
```

## 🚀 快速开始

### 方式1：使用配置文件（推荐）⭐

**训练模式**：

```bash
# 1. 激活环境（必须）
conda activate UAV_PATH_PLANNING

# 2. 使用默认配置训练
python scripts/baseline/train.py

# 3. 使用自定义配置（多Follower示例）
python scripts/baseline/train.py --config configs/masac/multi_follower.yaml
```

**测试模式**：

```bash
# 1. 激活环境（必须）
conda activate UAV_PATH_PLANNING

# 2. 使用默认配置测试
python scripts/baseline/test.py

# 3. 使用自定义配置测试
python scripts/baseline/test.py --config configs/masac/multi_follower.yaml
```

**修改配置**：

编辑 `configs/masac/default.yaml` 文件：

```yaml
# 环境配置
environment:
  n_leader: 1              # Leader数量
  n_follower: 1            # Follower数量
  render: false            # 是否渲染

# 训练配置
training:
  ep_max: 500              # 最大训练轮数
  ep_len: 1000             # 每轮最大步数
  gamma: 0.9               # 折扣因子
  batch_size: 128          # 批次大小

# 网络配置
network:
  hidden_dim: 256          # 隐藏层维度
  q_lr: 3.0e-4             # Q网络学习率
  policy_lr: 1.0e-3        # Policy学习率
```

详细配置说明见：[configs/README.md](configs/README.md)

### 方式2：直接修改脚本（旧方式）

修改 `scripts/baseline/train.py` 或 `configs/masac/default.yaml` 中的参数，然后运行训练或测试脚本。

## 📁 项目结构

```
UAV_PATH_PLANNING/
├── README.md                 # 项目文档
├── requirements.txt          # 项目依赖
├── .gitignore               # Git忽略规则
├── .cursor/                 # Cursor IDE配置
│   ├── rules/              # 项目开发规则
│   │   └── project-rules.mdc  # 核心开发规范（自动应用）
│   └── commands/           # 自定义斜杠命令
│       ├── init.md         # 初始化命令
│       └── ultrathink.md   # 深度思考模式命令
├── configs/                 # 配置文件目录
│   ├── README.md           # 配置文件说明
│   └── masac/              # MASAC算法配置
│       ├── default.yaml    # 默认配置
│       └── multi_follower.yaml  # 多Follower示例配置
├── scripts/                 # 训练和测试脚本
│   └── baseline/           # Baseline实验脚本
│       ├── train.py        # 训练脚本（支持配置文件）
│       └── test.py         # 测试脚本（支持配置文件）
├── algorithm/               # 算法实现
│   ├── __init__.py
│   └── masac/              # MASAC算法模块
│       ├── __init__.py     # 模块导出
│       ├── agent.py        # Actor、Critic、Entropy类
│       ├── model.py        # ActorNet、CriticNet网络
│       ├── buffer.py       # Memory经验回放
│       ├── noise.py        # OU噪声生成器
│       ├── trainer.py      # Trainer训练器类
│       └── tester.py       # Tester测试器类
├── rl_env/                  # 强化学习环境
│   ├── __init__.py
│   └── path_env.py         # 路径规划环境实现
├── assignment/              # 仿真组件
│   ├── components/         # 游戏组件
│   │   ├── player.py       # 无人机、障碍物等实体
│   │   └── info.py         # 信息显示
│   ├── constants.py        # 常量定义
│   ├── tools.py            # 工具函数
│   └── source/             # 资源文件
│       ├── image/          # 图片资源
│       └── music/          # 音效资源
└── utils/                   # 工具函数
    ├── __init__.py
    ├── path_utils.py       # 路径管理工具（自动处理跨平台路径）
    └── config_loader.py    # 配置文件加载工具
```

## ⚙️ 核心参数配置

### 设备配置（GPU加速）

| 参数名 | 默认值 | 说明 | 选项 |
|--------|--------|------|------|
| `device` | 'auto' | 训练设备 | 'auto', 'cpu', 'cuda', 'cuda:0' |

### 随机种子配置（可复现性）

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `seed` | 42 | 基础随机种子 |
| `deterministic` | false | 完全确定性模式（牺牲性能） |

**种子策略**：
- **训练种子**: `base_seed + episode` (例如: 42, 43, 44, ...)
- **测试种子**: `base_seed + 10000 + episode` (例如: 10042, 10043, 10044, ...)
- **种子分离**: 训练和测试使用完全不同的种子空间，避免过拟合
- **可复现性**: 相同的seed配置保证完全相同的训练结果

**使用示例**：
```yaml
# configs/masac/default.yaml
training:
  seed: 42                 # 修改此值改变实验的随机性
  deterministic: false     # 设为true启用完全确定性（慢10-30%）
```

**设备选项说明**：
- `'auto'`：自动检测GPU，有则用GPU，无则用CPU（**推荐**）
- `'cpu'`：强制使用CPU训练
- `'cuda'`：使用第一块GPU
- `'cuda:0'`, `'cuda:1'`：指定GPU编号（多GPU环境）

**性能提升**：
- 🚀 使用GPU训练速度提升 **2-10倍**（取决于GPU型号）
- 💾 环境采样保持在CPU（numpy操作）
- ⚡ 神经网络训练在GPU（梯度计算加速）

**GPU使用示例**：
```yaml
# configs/masac/default.yaml
training:
  device: 'auto'           # 自动检测GPU（推荐）
  # device: 'cuda'         # 强制使用GPU
  # device: 'cuda:1'       # 使用第2块GPU
  # device: 'cpu'          # 强制使用CPU
```

```bash
# 查看是否使用GPU
python scripts/baseline/train.py
# 输出: 🚀 使用GPU训练: NVIDIA GeForce RTX 3090 (24.0GB)
```

**显存需求**：
- 1 Leader + 1 Follower: ~50MB
- 1 Leader + 3 Followers: ~70MB
- 1 Leader + 10 Followers: ~150MB
- 入门级GPU（2GB显存）完全足够！

### 训练参数

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `EP_MAX` | 500 | 最大训练轮数 |
| `EP_LEN` | 1000 | 每轮最大步数 |
| `GAMMA` | 0.9 | 折扣因子 |
| `BATCH` | 128 | 批次大小 |
| `MemoryCapacity` | 20000 | 经验池容量 |

### 网络参数

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `q_lr` | 3e-4 | Q网络学习率 |
| `value_lr` | 3e-3 | Value网络学习率 |
| `policy_lr` | 1e-3 | Policy网络学习率 |
| `tau` | 1e-2 | 软更新系数 |

### 环境参数

| 参数名 | 默认值 | 说明 | 可配置范围 |
|--------|--------|------|----------|
| `N_LEADER` | 1 | Leader数量 | 目前支持1 |
| `N_FOLLOWER` | 1 | Follower数量 | **1-10（推荐）** |
| `state_number` | 7 | 状态维度 | 固定 |
| `action_number` | 2 | 动作维度 | 固定 |

**🎯 多 Follower 配置示例**：
```python
# scripts/baseline/train.py
N_LEADER = 1
N_FOLLOWER = 3  # 设置3个Follower
```

**⚠️ 性能建议**：
- 1-3 个 Follower：性能最佳，训练收敛快
- 4-6 个 Follower：性能良好，适合复杂编队
- 7-10 个 Follower：性能可接受，训练时间较长
- 10+ 个 Follower：可能出现性能瓶颈和收敛困难

## 🎯 算法原理

### SAC (Soft Actor-Critic)

SAC是一种基于最大熵的强化学习算法，具有以下特点：

- **最大熵框架**：在最大化累积奖励的同时最大化策略熵
- **Off-policy学习**：提高样本利用效率
- **连续动作空间**：适合无人机控制问题
- **稳定训练**：使用Double Q-learning和目标网络

### 网络架构

- **Actor网络**：输出动作的均值和标准差
- **Critic网络**：双Q网络架构，减少过估计
- **自动调节温度系数**：动态平衡探索与利用

### 奖励函数设计

```python
总奖励 = 边界奖励 + 避障奖励 + 目标奖励 + 编队奖励 + 速度奖励
```

- **边界奖励**：防止无人机飞出边界
- **避障奖励**：根据与障碍物距离调整
- **目标奖励**：引导无人机接近目标
- **编队奖励**：保持Leader-Follower队形
- **速度奖励**：鼓励速度匹配

## 📊 训练结果

训练完成后，系统会自动保存：

- **神经网络模型**：`leader.pth`（Leader）、`follower.pth`（Follower）
- **训练数据**：`data/MASAC_new1.pkl`（奖励统计数据）
- **可视化图表**：自动保存到 `plots/` 目录
  - `plots/total_reward.png` - 总奖励曲线
  - `plots/leader_reward.png` - Leader奖励曲线
  - `plots/follower_reward.png` - Follower奖励曲线

**注意**：每个Follower有独立的权重，但所有权重打包在一个 `follower.pth` 文件中

### 评估指标

测试模式下会计算以下指标：

- **任务完成率**：成功到达目标的比例
- **平均编队保持率**：保持编队的时间占比
- **平均飞行时间**：完成任务所需的平均步数
- **平均飞行路程**：累积飞行距离
- **平均能量损耗**：控制输入的累积值

## 🎮 可视化界面

启用可视化后（`RENDER=True`），界面显示：

- **蓝色飞机**：Leader无人机（负责路径规划和目标导航）
- **绿色飞机**：Follower无人机（跟随Leader保持编队）
- **黑色圆形**：障碍物（碰撞区域）
- **红色圆形**：目标点
- **蓝色轨迹线**：Leader飞行轨迹
- **绿色轨迹线**：Follower飞行轨迹

## 🤖 Cursor 自定义命令

本项目配置了 Cursor IDE 的自定义命令，提升开发效率：

### `/init` - 项目深度分析

当需要全面了解项目时，使用此命令让 AI 进行系统化的项目分析。

**适用场景**：
- 新成员快速了解项目
- 项目交接和文档整理
- 代码审查前的准备
- 重构前的全局把握
- 技术债务评估
- 项目健康度检查

**使用方法**：
```
/init
```

AI 将进行 **10 个维度**的深度分析：
1. 项目概览扫描（README、结构、配置）
2. 架构与技术栈分析（模块划分、依赖关系）
3. 核心代码深度阅读（算法、数据结构）
4. 配置与参数分析（超参数、环境变量）
5. 数据流与状态管理（输入输出、状态转换）
6. 测试与质量保证（测试覆盖、代码质量）
7. 依赖与环境分析（第三方库、兼容性）
8. 文档与注释分析（完整性、一致性）
9. Git 历史与演进分析（提交记录、活跃度）
10. 扩展性与未来规划（重构机会、改进建议）

**输出报告**包含：
- 📊 项目概览
- 🏗️ 架构分析
- 💻 核心代码分析
- ⚙️ 配置与参数
- 📈 数据流分析
- ✅ 质量评估
- 🔗 依赖分析
- 📚 文档评估
- 🚀 改进建议
- 🎓 学习路径

### `/ultrathink` - 深度思考模式

当遇到复杂问题时，使用此命令让 AI 进行更深入的分析和推理。

**适用场景**：
- 架构设计决策
- 复杂算法选型
- 性能优化问题
- 疑难 Bug 排查
- 重构方案制定
- 技术选型评估

**使用方法**：
```
/ultrathink 如何优化SAC算法的训练速度？
```

AI 将从以下 **6 个维度**进行深度分析：
1. 问题理解与分解
2. 多角度分析（技术、架构、安全、用户体验、长期影响）
3. 方案探索（列举多种方案并对比优劣）
4. 边缘案例思考
5. 深度推理（根因分析、副作用预测）
6. 最佳实践对照

### 项目开发规则

项目遵循严格的开发规范（`.cursor/rules/project-rules.mdc`），自动应用于所有开发过程：
- ✅ 每次修改后自动更新 README.md
- ✅ 提交前必须询问用户确认
- ✅ 使用 `git add .` 并遵守 .gitignore
- ✅ 提交后自动推送到 GitHub
- ✅ 保持简洁一致的提交信息

## 📂 模型保存

训练完成后，自动保存到 `saved_models/` 目录：

### 模型文件
- `leader.pth` - Leader 模型（包含所有Leader的权重）
- `follower.pth` - Follower 模型（包含所有Follower的权重）

### 训练数据
- `data/MASAC_new1.pkl` - 训练统计数据

### 奖励曲线图
- `plots/total_reward.png` - 总奖励曲线
- `plots/leader_reward.png` - Leader奖励曲线
- `plots/follower_reward.png` - Follower奖励曲线

**模型文件结构**：
- `leader.pth` 是一个字典，包含所有Leader的独立权重（leader_0, leader_1, ...）
- `follower.pth` 是一个字典，包含所有Follower的独立权重（follower_0, follower_1, ...）
- 每个智能体训练时有独立的权重，但保存在同一个文件中便于管理
- 无论有多少Follower，只需要2个模型文件：leader.pth 和 follower.pth

**建议**：将 `saved_models/` 目录添加到 `.gitignore` 以避免提交大文件。

## 🤖 MASAC 算法模块

项目将 SAC 算法封装为独立的 `algorithm/masac` 模块，提供模块化的强化学习组件。

### 模块结构

```python
from algorithm.masac import (
    Actor,              # 策略网络（Actor）
    Critic,             # 价值网络（Critic）
    Entropy,            # 熵调节器
    Memory,             # 经验回放缓冲区
    ActorNet,           # Actor神经网络
    CriticNet,          # Critic神经网络
    Ornstein_Uhlenbeck_Noise,  # OU噪声
    Trainer,            # 训练器类（封装完整训练流程）
    Tester              # 测试器类（封装完整测试流程）
)
```

### 使用示例

#### 方式1：使用 Trainer/Tester 类（推荐）

```python
from rl_env.path_env import RlGame
from algorithm.masac import Trainer, Tester

# 创建环境
env = RlGame(n=1, m=1, render=False).unwrapped

# 训练
trainer = Trainer(
    env=env,
    n_leader=1,
    n_follower=1,
    state_dim=7,
    action_dim=2,
    max_action=1.0,
    min_action=-1.0,
    hidden_dim=256,
    gamma=0.9,
    q_lr=3e-4,
    value_lr=3e-3,
    policy_lr=1e-3,
    tau=1e-2,
    batch_size=128,
    memory_capacity=20000
)
trainer.train(ep_max=500, ep_len=1000, render=False)

# 测试
tester = Tester(
    env=env,
    n_leader=1,
    n_follower=1,
    state_dim=7,
    action_dim=2,
    max_action=1.0,
    min_action=-1.0,
    hidden_dim=256,
    policy_lr=1e-3
)
results = tester.test(ep_len=1000, test_episode=100, render=False)
```

#### 方式2：使用底层组件（高级用法）

```python
import numpy as np
from algorithm.masac import Actor, Critic, Entropy, Memory, Ornstein_Uhlenbeck_Noise

# 创建 Actor（所有参数必须显式传入）
actor = Actor(
    state_dim=7,
    action_dim=2,
    max_action=1.0,
    min_action=-1.0,
    hidden_dim=256,
    policy_lr=1e-3
)

# 创建 Critic（所有参数必须显式传入）
critic = Critic(
    state_dim=14,  # state_dim * (N_LEADER + N_FOLLOWER)
    action_dim=2,
    hidden_dim=256,
    value_lr=3e-3,
    tau=1e-2
)

# 创建 Entropy 调节器（所有参数必须显式传入）
entropy = Entropy(
    target_entropy=-0.1,
    lr=3e-4
)

# 创建经验回放缓冲区
state_dim = 7
action_dim = 2
num_agents = 2
transition_dim = 2 * state_dim * num_agents + action_dim * num_agents + num_agents
memory = Memory(
    capacity=20000,
    transition_dim=transition_dim
)

# 创建 OU 噪声生成器（所有参数必须显式传入）
noise = Ornstein_Uhlenbeck_Noise(
    mean=np.zeros((num_agents, action_dim)),
    sigma=0.1,
    theta=0.1,
    dt=1e-2
)

# 选择动作
state = np.random.randn(7)
action = actor.choose_action(state)

# 评估动作（用于训练）
action, log_prob = actor.evaluate(state)
```

### 命名规范

所有类和方法都采用清晰明了的命名：

**类和参数**：
- `state_dim`, `action_dim` 替代 `inp`, `outp`
- `hidden_dim` 替代隐式的256
- `fc1`, `fc2` 替代 `in_to_y1`, `y1_to_y2`
- `mean_layer`, `log_std_layer` 替代 `out`, `std_out`

**Critic 网络**：
- `critic_net` 替代 `critic_v`
- `get_q_value()` 替代 `get_v()`
- `loss_fn` 替代 `lossfunc`

**Memory 缓冲区**：
- `buffer` 替代 `mem`
- `counter` 替代 `memory_counter`
- `store()` 替代 `store_transition()`
- `transition_dim` 替代 `dims`

**OU Noise**：
- `mean` 替代 `mu`
- `current_noise` 替代 `x_prev`
- `initial_noise` 替代 `x0`

### 设计原则

**参数显式化**：
- ⚠️ 所有 `__init__` 方法的参数都**没有默认值**
- ✅ 必须显式传入每个参数，便于调试和追踪
- ✅ 避免隐式默认值导致的问题
- ✅ 代码更加明确和可维护

**示例对比**：
```python
# ❌ 隐式默认值（不推荐）
actor = Actor(state_dim=7, action_dim=2)  # 其他参数使用默认值

# ✅ 显式传参（推荐，当前实现）
actor = Actor(
    state_dim=7,
    action_dim=2,
    max_action=1.0,
    min_action=-1.0,
    hidden_dim=256,
    policy_lr=1e-3
)
```

### 模块优势

- ✅ **参数显式化**：所有参数必须显式传入，便于调试
- ✅ **清晰命名**：所有变量和方法名语义明确
- ✅ **模块化设计**：各组件独立，易于维护和测试
- ✅ **参数化配置**：所有超参数通过构造函数传入
- ✅ **无全局依赖**：不依赖外部全局变量
- ✅ **文档完整**：每个类和方法都有文档字符串
- ✅ **易于扩展**：可快速替换不同的网络结构
- ✅ **代码复用**：可在其他项目中直接使用

## 🛠️ 路径管理工具

项目内置了跨平台路径管理工具 `utils/path_utils.py`，确保项目可以在任何设备上运行。

### 主要功能

```python
from utils import (
    get_project_root,
    get_abs_path,
    get_model_path,
    get_data_path,
    get_resource_path,
    ensure_dir
)

# 获取项目根目录
root = get_project_root()

# 相对路径转绝对路径
config_path = get_abs_path('config/settings.json')

# 获取模型保存路径（自动创建目录）
model_path = get_model_path('my_model.pth')

# 获取数据文件路径
data_path = get_data_path('train_data.pkl')

# 获取资源文件路径（图片、音频等）
image_path = get_resource_path('image')
music_path = get_resource_path('music')

# 确保目录存在
save_dir = ensure_dir('my_custom_dir')
```

### 常用路径变量

```python
from utils.path_utils import PROJECT_ROOT, SAVE_DIR, LOG_DIR, DATA_DIR

print(PROJECT_ROOT)  # 项目根目录
print(SAVE_DIR)      # saved_models/
print(LOG_DIR)       # saved_models/logs/
print(DATA_DIR)      # saved_models/data/
```

### 优势

- ✅ **跨平台兼容**：自动处理 Windows/Linux/macOS 路径差异
- ✅ **相对路径**：基于项目根目录，无需硬编码
- ✅ **自动创建**：目录不存在时自动创建
- ✅ **易于迁移**：项目可在任何设备直接运行

## 🔧 故障排除

### 常见问题

1. **多 Follower 支持**
   - ✅ **已支持**：可以配置任意数量的 Follower（建议 ≤ 10）
   - 修改 `scripts/baseline/train.py` 中的 `N_FOLLOWER` 参数即可
   - 示例：`N_FOLLOWER = 3` 将创建 3 个 Follower 无人机

2. **依赖安装问题**
   - 确保安装的是 `gymnasium` 而不是旧的 `gym`
   - 如果之前安装过 gym，建议先卸载：`pip uninstall gym`
   - 然后安装 gymnasium：`pip install gymnasium>=0.28.0`

3. **MSELoss 维度不匹配警告**
   - 已修复：使用 `log_prob.sum(dim=-1, keepdim=True)` 确保维度匹配

4. **pygame初始化失败**
   - 确保安装了pygame：`pip install pygame`
   - 如果无需可视化，设置 `RENDER=False`

### 最近更新 (2025-10-28)

#### 🎲 固定种子系统（最新）
✅ **完全可复现**：相同seed配置保证完全相同的训练结果  
✅ **Episode级种子**：每个Episode使用不同种子，探索多样性  
✅ **种子空间分离**：训练和测试使用不同种子空间（+10000偏移）  
✅ **多库同步**：Python/NumPy/PyTorch/CUDA种子统一管理  
✅ **确定性选项**：可选择完全确定性模式（牺牲性能）  
✅ **配置化管理**：通过YAML配置基础种子  

#### 🚀 GPU加速训练支持
✅ **自动设备检测**：自动检测并使用GPU，无GPU时降级到CPU  
✅ **配置化设备选择**：通过YAML配置文件指定训练设备  
✅ **CPU-GPU混合**：环境采样在CPU，神经网络训练在GPU  
✅ **智能模型保存**：保存时自动转CPU，加载时自动映射到目标设备  
✅ **性能提升显著**：GPU训练速度提升2-10倍  
✅ **多GPU支持**：可指定GPU编号（cuda:0, cuda:1）  
✅ **零代码修改**：修改配置文件即可启用GPU  

#### 🎉 YAML配置文件管理
✅ **配置文件化**：所有参数通过YAML文件管理  
✅ **命令行支持**：支持 --config 参数指定配置文件  
✅ **多配置管理**：支持创建多个实验配置  
✅ **参数集中**：无需修改代码，只需编辑配置文件  

#### 📊 优化训练输出格式
✅ **表格化输出**：每个Episode一行，清晰展示所有信息  
✅ **分别显示奖励**：Leader和每个Follower的奖励独立统计  
✅ **步数统计**：显示每轮实际使用的步数  
✅ **状态标识**：带颜色的状态（Success/Failure/Timeout）  
✅ **动态适应**：自动适应不同数量的Follower  

#### 🎉 支持可配置 Follower 数量
✅ **解除硬编码限制**：移除 N_FOLLOWER=1 的强制约束  
✅ **动态状态构建**：reset() 方法支持任意数量 Follower  
✅ **循环奖励计算**：step() 方法自动处理多个 Follower 的编队奖励  
✅ **多轨迹可视化**：不同颜色区分不同 Follower 的飞行轨迹  
✅ **向后兼容**：N_FOLLOWER=1 时行为与原版完全一致  
✅ **编队智能化**：Leader 同时考虑所有 Follower 的编队状态  

#### 重大重构：Trainer/Tester 类封装 🎉
✅ **提取训练器类**：新增 `algorithm/masac/trainer.py` - Trainer 类（445行）  
✅ **提取测试器类**：新增 `algorithm/masac/tester.py` - Tester 类（226行）  
✅ **独立训练脚本**：新增 `scripts/baseline/train.py` - 独立训练入口（125行）  
✅ **独立测试脚本**：新增 `scripts/baseline/test.py` - 独立测试入口（106行）  
✅ **职责分离设计**：配置参数（__init__）与运行参数（train/test）完全分离  
✅ **移除 main_SAC.py**：使用独立脚本替代单文件 Switch 控制模式  
✅ **代码精简优化**：核心逻辑从 383 行优化为模块化设计  
✅ **易于批量实验**：可创建多个训练脚本对比不同配置  
✅ **提升可测试性**：Trainer/Tester 类可独立进行单元测试  
✅ **模块化程度提升**：从单文件设计升级为完整的类封装设计  

#### 之前的更新
✅ **命名风格统一**：M_FOLLOWER → N_FOLLOWER，与 N_LEADER 保持一致  
✅ **参数显式化设计**：移除所有默认参数值，强制显式传参便于调试  
✅ **MASAC 模块命名规范化**：所有变量、方法名清晰明了  
✅ **MASAC 模块化封装**：算法组件独立为可复用模块  
✅ **重构命名体系**：Hero/Enemy → Leader/Follower，更符合学术规范  
✅ **新增路径管理工具**：`utils/path_utils.py` 实现跨平台路径管理  
✅ **升级到 Gymnasium**：替换已过时的 Gym 库  
✅ 修复 MSELoss 维度不匹配问题（SAC 算法 bug）  
✅ 移除所有硬编码绝对路径，使用智能路径工具  
✅ 自动创建 `saved_models/` 目录保存模型  
✅ 参数化配置，无全局变量依赖  
✅ 完整的文档字符串和注释  
✅ 消除代码重复，遵循 DRY 原则  
✅ 支持项目在任何设备直接运行  
✅ 代码更加规范，易于论文发表和分享

### 迁移说明

如果您之前使用的是 Gym：
```bash
# 1. 卸载旧的 gym
pip uninstall gym

# 2. 安装 gymnasium
pip install gymnasium>=0.28.0

# 3. 代码已自动适配，无需手动修改
```

## 📝 开发规范

本项目遵循严格的开发规范，详见 `.cursor/rules/project-rules.mdc`

核心原则：
- 遵循PEP 8代码规范
- 禁止硬编码绝对路径
- 使用 `git add .` 提交代码
- 保持提交信息简洁一致

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'feat: 添加某个特性'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 📧 联系方式

- 项目维护者：[Your Name]
- 项目链接：[https://github.com/your-username/UAV_PATH_PLANNING](https://github.com/your-username/UAV_PATH_PLANNING)
- 问题反馈：[Issues](https://github.com/your-username/UAV_PATH_PLANNING/issues)

## 🙏 致谢

感谢所有为本项目做出贡献的开发者！

---

**最后更新时间**: 2025-10-28  
**版本**: v1.0.0

