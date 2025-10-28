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

### 训练模式

修改 `main_SAC.py` 中的参数：

```python
Switch = 0  # 0为训练模式
RENDER = False  # 训练时建议关闭渲染
N_LEADER = 1  # Leader数量
M_FOLLOWER = 1  # Follower数量
```

运行训练：

```bash
# 1. 激活环境（必须）
conda activate UAV_PATH_PLANNING

# 2. 运行训练
python main_SAC.py
```

### 测试模式

修改 `main_SAC.py` 中的参数：

```python
Switch = 1  # 1为测试模式
RENDER = True  # 测试时可开启可视化
TEST_EPIOSDE = 100  # 测试轮数
```

运行测试：

```bash
# 1. 激活环境（必须）
conda activate UAV_PATH_PLANNING

# 2. 运行测试
python main_SAC.py
```

## 📁 项目结构

```
UAV_PATH_PLANNING/
├── README.md                 # 项目文档
├── requirements.txt          # 项目依赖
├── .gitignore               # Git忽略规则
├── main_SAC.py              # 主程序入口
├── .cursor/                 # Cursor IDE配置
│   ├── rules/              # 项目开发规则
│   │   └── project-rules.mdc  # 核心开发规范（自动应用）
│   └── commands/           # 自定义斜杠命令
│       ├── init.md         # 初始化命令
│       └── ultrathink.md   # 深度思考模式命令
├── algorithm/               # 算法实现
│   ├── __init__.py
│   └── masac/              # MASAC算法模块
│       ├── __init__.py     # 模块导出
│       ├── agent.py        # Actor、Critic、Entropy类
│       ├── model.py        # ActorNet、CriticNet网络
│       ├── buffer.py       # Memory经验回放
│       └── noise.py        # OU噪声生成器
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
├── utils/                   # 工具函数
│   ├── __init__.py
│   └── path_utils.py       # 路径管理工具（自动处理跨平台路径）
└── docs/                    # 文档目录
```

## ⚙️ 核心参数配置

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

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `N_LEADER` | 1 | Leader数量 |
| `M_FOLLOWER` | 1 | Follower数量 |
| `state_number` | 7 | 状态维度 |
| `action_number` | 2 | 动作维度 |

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

- 神经网络模型：`Path_SAC_actor_L1.pth`（Leader）、`Path_SAC_actor_F1.pth`（Follower）
- 训练数据：包含奖励曲线的pickle文件
- 可视化图表：Total reward、Leader reward、Follower reward曲线

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

训练过程中，模型会自动保存到 `saved_models/` 目录：
- `Path_SAC_actor_L1.pth` - Leader 模型
- `Path_SAC_actor_F1.pth` - Follower 模型
- `MASAC_new1.pkl` - 训练数据

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
    Ornstein_Uhlenbeck_Noise  # OU噪声
)
```

### 使用示例

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
    state_dim=14,  # state_dim * (N_LEADER + M_FOLLOWER)
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

1. **M_FOLLOWER必须为1的错误**
   - 当前版本仅支持1个Follower
   - 需要修改 `path_env.py` 中的相关代码以支持多个Follower

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

