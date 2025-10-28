# UAV路径规划系统 (UAV Path Planning)

基于深度强化学习（SAC算法）的多智能体无人机协同路径规划系统

## 📖 项目简介

本项目实现了一个基于Soft Actor-Critic (SAC)算法的无人机路径规划系统，支持多智能体协同、动态避障和目标追踪。系统使用PyTorch构建深度神经网络，通过强化学习训练智能体在复杂环境中完成路径规划任务。

### 主要特性

- ✅ **多智能体协同**：支持Leader-Follower编队控制
- ✅ **动态避障**：实时检测并规避障碍物
- ✅ **目标导航**：引导无人机到达指定目标点
- ✅ **可视化仿真**：基于Pygame的实时可视化
- ✅ **自动训练**：支持离线训练和在线测试
- ✅ **模型保存**：自动保存训练好的神经网络模型

## 🛠️ 技术栈

- **深度学习框架**：PyTorch
- **强化学习环境**：OpenAI Gym
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

# 安装依赖包（方式1：使用 requirements.txt）
pip install -r requirements.txt

# 或者手动安装（方式2）
pip install torch torchvision
pip install gym pygame numpy matplotlib
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
N_Agent = 1  # 智能体数量
M_Enemy = 1  # 跟随者数量
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
│   └── masac/              # MASAC算法相关
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
| `N_Agent` | 1 | Leader数量 |
| `M_Enemy` | 1 | Follower数量 |
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

- 蓝色：Leader无人机
- 绿色：Follower无人机
- 黑色圆形：障碍物（碰撞区域）
- 红色圆形：目标点
- 彩色轨迹线：无人机飞行轨迹

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

## 🔧 故障排除

### 常见问题

1. **M_Enemy必须为1的错误**
   - 当前版本仅支持1个Follower
   - 需要修改 `path_env.py` 中的相关代码以支持多个Follower

2. **路径找不到错误**
   - 检查硬编码的绝对路径
   - 修改为相对路径或使用 `os.path.join()`

3. **pygame初始化失败**
   - 确保安装了pygame：`pip install pygame`
   - 如果无需可视化，设置 `RENDER=False`

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

