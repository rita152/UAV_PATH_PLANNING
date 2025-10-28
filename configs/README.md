# 配置文件说明

## 📁 目录结构

```
configs/
├── README.md                    # 本文档
└── masac/                       # MASAC算法配置
    ├── default.yaml            # 默认配置（1 Leader + 1 Follower）
    └── multi_follower.yaml     # 多Follower示例（1 Leader + 3 Followers）
```

## 🎯 使用方法

### 1. 使用默认配置

```bash
# 训练
conda activate UAV_PATH_PLANNING
python scripts/baseline/train.py

# 测试
python scripts/baseline/test.py
```

### 2. 使用自定义配置

```bash
# 使用多Follower配置训练
python scripts/baseline/train.py --config configs/masac/multi_follower.yaml

# 使用多Follower配置测试
python scripts/baseline/test.py --config configs/masac/multi_follower.yaml
```

### 3. 创建自己的配置

复制并修改现有配置文件：

```bash
cp configs/masac/default.yaml configs/masac/my_config.yaml
# 编辑 my_config.yaml
python scripts/baseline/train.py --config configs/masac/my_config.yaml
```

## 📋 配置文件结构

### environment（环境配置）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_leader` | int | 1 | Leader数量 |
| `n_follower` | int | 1 | Follower数量 |
| `render` | bool | false | 是否渲染可视化 |
| `state_dim` | int | 7 | 状态维度 |
| `action_dim` | int | 2 | 动作维度 |

### training（训练配置）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ep_max` | int | 500 | 最大训练轮数 |
| `ep_len` | int | 1000 | 每轮最大步数 |
| `train_num` | int | 1 | 训练次数（重复实验） |
| `gamma` | float | 0.9 | 折扣因子 |
| `batch_size` | int | 128 | 批次大小 |
| `memory_capacity` | int | 20000 | 经验池容量 |
| `data_save_name` | str | 'MASAC_new1.pkl' | 数据保存文件名 |

### testing（测试配置）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `test_episode` | int | 100 | 测试轮数 |
| `ep_len` | int | 1000 | 每轮最大步数 |
| `render` | bool | false | 是否渲染可视化 |
| `leader_model_path` | str/null | null | Leader模型路径（null=默认） |
| `follower_model_path` | str/null | null | Follower模型路径（null=默认） |

### network（网络配置）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hidden_dim` | int | 256 | 隐藏层维度 |
| `q_lr` | float | 3.0e-4 | Q网络学习率 |
| `value_lr` | float | 3.0e-3 | Value网络学习率 |
| `policy_lr` | float | 1.0e-3 | Policy网络学习率 |
| `tau` | float | 1.0e-2 | 软更新系数 |

### env_vars（环境变量）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `KMP_DUPLICATE_LIB_OK` | str | 'TRUE' | Intel MKL兼容性 |

### output（输出配置）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `verbose` | bool | true | 是否输出详细信息 |
| `log_interval` | int | 1 | 日志输出间隔 |
| `save_interval` | int | 20 | 模型保存间隔 |

## 💡 常用配置示例

### 快速测试（减少训练轮数）

```yaml
training:
  ep_max: 50              # 仅训练50轮
  ep_len: 500             # 每轮500步
  train_num: 1
  # ... 其他参数保持不变
```

### 多Follower编队（3个Follower）

```yaml
environment:
  n_leader: 1
  n_follower: 3           # 3个Follower
  # ... 其他参数保持不变

training:
  data_save_name: 'MASAC_multi_follower_3.pkl'  # 修改保存文件名
```

### 可视化训练

```yaml
environment:
  render: true            # 开启可视化
  # ... 其他参数保持不变

training:
  ep_max: 10              # 减少轮数（可视化较慢）
  ep_len: 200
```

### 调整学习率

```yaml
network:
  q_lr: 1.0e-4            # 降低学习率
  value_lr: 1.0e-3
  policy_lr: 5.0e-4
  # ... 其他参数保持不变
```

## 🎓 最佳实践

### 1. 命名规范

配置文件建议命名格式：`<实验名称>_<特征>.yaml`

示例：
- `baseline.yaml` - 基准配置
- `multi_follower_5.yaml` - 5个Follower配置
- `large_batch.yaml` - 大批次训练
- `high_lr.yaml` - 高学习率实验

### 2. 版本管理

将配置文件纳入Git版本控制：

```bash
git add configs/masac/my_experiment.yaml
git commit -m "feat: 添加实验配置my_experiment"
```

### 3. 实验对比

为每个实验创建独立配置，便于对比：

```
configs/masac/
├── baseline.yaml
├── experiment_1_high_lr.yaml
├── experiment_2_large_batch.yaml
└── experiment_3_multi_follower.yaml
```

### 4. 参数搜索

通过修改配置文件进行超参数搜索：

```bash
# 实验1：baseline
python scripts/baseline/train.py --config configs/masac/baseline.yaml

# 实验2：高学习率
python scripts/baseline/train.py --config configs/masac/high_lr.yaml

# 实验3：大批次
python scripts/baseline/train.py --config configs/masac/large_batch.yaml
```

## 🔧 故障排除

### 配置文件找不到

确保配置文件路径正确：

```bash
# 从项目根目录运行
cd /path/to/UAV_PATH_PLANNING
python scripts/baseline/train.py --config configs/masac/my_config.yaml
```

### YAML格式错误

检查YAML格式是否正确：

```bash
# 使用Python验证YAML
python -c "import yaml; yaml.safe_load(open('configs/masac/my_config.yaml'))"
```

### 参数类型错误

确保数值类型正确：

```yaml
# ❌ 错误
ep_max: "500"           # 字符串

# ✅ 正确
ep_max: 500             # 整数

# ✅ 正确（科学计数法）
q_lr: 3.0e-4            # 浮点数
```

## 📝 配置文件模板

创建新配置时，复制以下模板：

```yaml
# 配置名称：<实验名称>
# 用途：<实验目的>
# 作者：<你的名字>
# 日期：<创建日期>

environment:
  n_leader: 1
  n_follower: 1
  render: false
  state_dim: 7
  action_dim: 2

training:
  ep_max: 500
  ep_len: 1000
  train_num: 1
  gamma: 0.9
  batch_size: 128
  memory_capacity: 20000
  data_save_name: 'experiment_name.pkl'

testing:
  test_episode: 100
  ep_len: 1000
  render: false
  leader_model_path: null
  follower_model_path: null

network:
  hidden_dim: 256
  q_lr: 3.0e-4
  value_lr: 3.0e-3
  policy_lr: 1.0e-3
  tau: 1.0e-2

env_vars:
  KMP_DUPLICATE_LIB_OK: 'TRUE'

output:
  verbose: true
  log_interval: 1
  save_interval: 20
```

---

**更新日期**: 2025-10-28  
**版本**: v1.0

