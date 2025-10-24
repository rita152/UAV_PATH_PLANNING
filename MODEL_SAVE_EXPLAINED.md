# 模型权重保存机制完整说明

## 📋 保存机制概述

模型权重文件会在训练过程中自动保存，以便后续测试或恢复训练。

**新的保存策略**: 
- 所有Leader保存为1个文件: `leader.pth`
- 所有Follower保存为1个文件: `follower.pth`
- 无论有多少个follower，都只生成2个文件

---

## ⚙️ 保存配置

### 配置文件设置

在 `configs/masac/default.yaml` 中：

```yaml
output:
  save_interval: 20     # 模型保存间隔（轮数）
  save_threshold: 200   # 开始保存模型的最小轮数
  output_dir: "output"  # 输出目录
```

### 参数说明

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `save_interval` | 20 | 每隔多少轮保存一次模型 |
| `save_threshold` | 200 | 超过多少轮后才开始保存（早期训练质量低，不保存） |
| `output_dir` | "output" | 模型保存的目录 |

---

## 🕐 保存时机

### 判断逻辑

```python
if episode % save_interval == 0 and episode > save_threshold:
    保存模型
```

### 实际保存的Episode

**示例**（默认配置：interval=20, threshold=200）：

| Episode | 是否保存 | 原因 |
|---------|---------|------|
| 20 | ❌ | 20 ≤ 200 (未达到阈值) |
| 100 | ❌ | 100 ≤ 200 (未达到阈值) |
| 200 | ❌ | 200 ≤ 200 (必须 > 200) |
| 201 | ❌ | 201 % 20 ≠ 0 |
| 220 | ✅ | 220 > 200 且 220 % 20 = 0 |
| 240 | ✅ | 240 > 200 且 240 % 20 = 0 |
| 260 | ✅ | 260 > 200 且 260 % 20 = 0 |
| ... | ... | ... |
| 480 | ✅ | 保存 |
| 500 | ✅ | 保存（最后一轮） |

**总结**：在500轮训练中，会保存约15次（从220轮开始，每20轮一次）

---

## 📁 保存的文件

### 文件命名规则（新机制）

```
output/
├── leader.pth        # 所有Leader（1个文件）
└── follower.pth      # 所有Follower（1个文件）
```

**优势**:
- ✅ 文件数量固定（只有2个）
- ✅ 易于管理和迁移
- ✅ follower数量改变时仍可部分加载
- ✅ 清晰的角色区分

**示例**:
- 1 leader + 3 followers → 2个文件
- 1 leader + 5 followers → 2个文件
- 1 leader + 10 followers → 2个文件

### 文件内容

**leader.pth 结构**:
```python
{
    'models': [
        {
            'net': leader_0_network_params,  # Leader 0的网络参数
            'opt': leader_0_optimizer_state   # Leader 0的优化器状态
        },
        # 如果有多个leader，继续添加...
    ],
    'n_leaders': 1  # Leader数量
}
```

**follower.pth 结构**:
```python
{
    'models': [
        {
            'net': follower_0_network_params,  # Follower 0的网络参数
            'opt': follower_0_optimizer_state   # Follower 0的优化器状态
        },
        {
            'net': follower_1_network_params,  # Follower 1的网络参数
            'opt': follower_1_optimizer_state   # Follower 1的优化器状态
        },
        # 更多follower...
    ],
    'n_followers': 2  # Follower数量（例如2个）
}
```

**包含内容**:
- `models`: 模型列表（所有leader或所有follower）
- `n_leaders` / `n_followers`: 数量信息（用于验证）

---

## 🔄 保存和加载流程

### 保存流程

```
训练循环
  ↓
每完成一个episode
  ↓
检查是否满足保存条件
  episode % save_interval == 0 AND episode > save_threshold
  ↓ 是
调用 save_models(output_dir)
  ↓
遍历所有actor
  ↓
保存每个actor的网络和优化器
  torch.save({'net': ..., 'opt': ...}, 'actor_i.pth')
```

### 加载流程

```
创建MASACTester
  ↓
调用 load_models(model_dir)
  ↓
遍历所有actor
  ↓
检查模型文件是否存在
  ↓ 存在
加载模型
  checkpoint = torch.load(path, map_location=device)
  actor.action_net.load_state_dict(checkpoint['net'])
```

---

## 🎯 自定义保存策略

### 1. 更频繁保存

```yaml
output:
  save_interval: 10     # 每10轮保存一次
  save_threshold: 100   # 100轮后开始保存
```

**效果**: 保存更频繁，但占用更多磁盘空间

### 2. 减少保存次数

```yaml
output:
  save_interval: 50     # 每50轮保存一次
  save_threshold: 300   # 300轮后开始保存
```

**效果**: 节省磁盘空间，但丢失中间状态

### 3. 从一开始就保存

```yaml
output:
  save_interval: 20
  save_threshold: 0     # 从第0轮开始
```

**效果**: 保存所有训练阶段的模型

---

## 📊 保存策略建议

### 推荐配置（不同场景）

#### 场景1: 正常训练（默认）

```yaml
output:
  save_interval: 20
  save_threshold: 200
```

**理由**:
- 前200轮模型质量较低，不需要保存
- 每20轮保存一次，平衡存储和安全性

#### 场景2: 调试模式

```yaml
output:
  save_interval: 10
  save_threshold: 0
```

**理由**:
- 频繁保存便于分析
- 从一开始就保存

#### 场景3: 长时间训练

```yaml
output:
  save_interval: 50
  save_threshold: 500
```

**理由**:
- 只保存后期收敛阶段的模型
- 节省磁盘空间

#### 场景4: 每轮都保存

```yaml
output:
  save_interval: 1
  save_threshold: 0
```

**理由**:
- 不错过任何一轮
- 用于详细分析训练过程

---

## 💾 磁盘空间估算

### 单个模型文件大小

**Actor网络** (hidden_dim=256):
- 参数量: ~200K
- 文件大小: ~1MB per actor

**示例**:
- 3个智能体: ~3MB per checkpoint
- 保存15次: ~45MB
- 500轮，每轮保存: ~1.5GB

---

## 🔧 代码实现

### save_models 方法

```python
def save_models(self, output_dir):
    """保存模型"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, actor in enumerate(self.actors):
        save_data = {
            'net': actor.action_net.state_dict(),  # 网络参数
            'opt': actor.optimizer.state_dict()    # 优化器状态
        }
        path = os.path.join(output_dir, f'actor_{i}.pth')
        torch.save(save_data, path)
    
    print(f"模型已保存到 {output_dir}")
```

### load_models 方法

```python
def load_models(self, output_dir):
    """加载模型"""
    for i, actor in enumerate(self.actors):
        path = os.path.join(output_dir, f'actor_{i}.pth')
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            actor.action_net.load_state_dict(checkpoint['net'])
            # 注意：测试时不加载优化器状态
```

---

## ⚠️ 注意事项

### 1. 只保存Actor网络

当前实现**只保存Actor**，不保存Critic和Entropy：

- ✅ Actor: 保存（用于测试）
- ❌ Critic: 不保存（仅训练时使用）
- ❌ Entropy: 不保存（仅训练时使用）

**原因**: 测试时只需要策略网络（Actor），不需要价值网络

### 2. 覆盖保存

每次保存会**覆盖**之前的文件：

```
第一次保存（Episode 220）: actor_0.pth, actor_1.pth, ...
第二次保存（Episode 240）: 覆盖 actor_0.pth, actor_1.pth, ...
```

**如果需要保留历史版本**，可以修改为：

```python
path = os.path.join(output_dir, f'actor_{i}_ep{episode}.pth')
```

### 3. GPU/CPU兼容

模型加载时支持设备转换：

```python
# GPU训练的模型可以在CPU上加载
checkpoint = torch.load(path, map_location=self.device)
```

---

## 🎓 使用示例

### 示例1: 使用默认保存设置

```bash
python scripts/masac/train.py --n_followers 3
```

**保存行为**:
- Episode 220, 240, 260, ..., 500: 保存模型

### 示例2: 自定义保存设置

```yaml
# 创建 configs/frequent_save.yaml
output:
  save_interval: 10     # 每10轮保存
  save_threshold: 50    # 50轮后开始
```

```bash
python scripts/masac/train.py --config configs/frequent_save.yaml
```

**保存行为**:
- Episode 60, 70, 80, ..., 500: 保存模型

### 示例3: 指定输出目录

```bash
python scripts/masac/train.py --n_followers 3 --output_dir output/exp1
```

**保存位置**: `output/exp1/actor_*.pth`

---

## 📝 总结

### 当前保存机制（已修正）

**配置参数** ✅:
```yaml
output:
  save_interval: 20     # 间隔
  save_threshold: 200   # 阈值
  output_dir: "output"  # 目录
```

**代码实现** ✅:
```python
if episode % self.save_interval == 0 and episode > self.save_threshold:
    self.save_models(output_dir)
```

**保存内容**:
- 每个Actor的网络参数
- 每个Actor的优化器状态

**保存格式**:
- PyTorch .pth 格式
- 文件名: `actor_{id}.pth`

**加载兼容**:
- ✅ GPU/CPU互通
- ✅ 自动设备映射

---

**已修正**: 配置参数现在正确使用！✨


## 📋 保存机制概述

模型权重文件会在训练过程中自动保存，以便后续测试或恢复训练。

**新的保存策略**: 
- 所有Leader保存为1个文件: `leader.pth`
- 所有Follower保存为1个文件: `follower.pth`
- 无论有多少个follower，都只生成2个文件

---

## ⚙️ 保存配置

### 配置文件设置

在 `configs/masac/default.yaml` 中：

```yaml
output:
  save_interval: 20     # 模型保存间隔（轮数）
  save_threshold: 200   # 开始保存模型的最小轮数
  output_dir: "output"  # 输出目录
```

### 参数说明

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `save_interval` | 20 | 每隔多少轮保存一次模型 |
| `save_threshold` | 200 | 超过多少轮后才开始保存（早期训练质量低，不保存） |
| `output_dir` | "output" | 模型保存的目录 |

---

## 🕐 保存时机

### 判断逻辑

```python
if episode % save_interval == 0 and episode > save_threshold:
    保存模型
```

### 实际保存的Episode

**示例**（默认配置：interval=20, threshold=200）：

| Episode | 是否保存 | 原因 |
|---------|---------|------|
| 20 | ❌ | 20 ≤ 200 (未达到阈值) |
| 100 | ❌ | 100 ≤ 200 (未达到阈值) |
| 200 | ❌ | 200 ≤ 200 (必须 > 200) |
| 201 | ❌ | 201 % 20 ≠ 0 |
| 220 | ✅ | 220 > 200 且 220 % 20 = 0 |
| 240 | ✅ | 240 > 200 且 240 % 20 = 0 |
| 260 | ✅ | 260 > 200 且 260 % 20 = 0 |
| ... | ... | ... |
| 480 | ✅ | 保存 |
| 500 | ✅ | 保存（最后一轮） |

**总结**：在500轮训练中，会保存约15次（从220轮开始，每20轮一次）

---

## 📁 保存的文件

### 文件命名规则（新机制）

```
output/
├── leader.pth        # 所有Leader（1个文件）
└── follower.pth      # 所有Follower（1个文件）
```

**优势**:
- ✅ 文件数量固定（只有2个）
- ✅ 易于管理和迁移
- ✅ follower数量改变时仍可部分加载
- ✅ 清晰的角色区分

**示例**:
- 1 leader + 3 followers → 2个文件
- 1 leader + 5 followers → 2个文件
- 1 leader + 10 followers → 2个文件

### 文件内容

**leader.pth 结构**:
```python
{
    'models': [
        {
            'net': leader_0_network_params,  # Leader 0的网络参数
            'opt': leader_0_optimizer_state   # Leader 0的优化器状态
        },
        # 如果有多个leader，继续添加...
    ],
    'n_leaders': 1  # Leader数量
}
```

**follower.pth 结构**:
```python
{
    'models': [
        {
            'net': follower_0_network_params,  # Follower 0的网络参数
            'opt': follower_0_optimizer_state   # Follower 0的优化器状态
        },
        {
            'net': follower_1_network_params,  # Follower 1的网络参数
            'opt': follower_1_optimizer_state   # Follower 1的优化器状态
        },
        # 更多follower...
    ],
    'n_followers': 2  # Follower数量（例如2个）
}
```

**包含内容**:
- `models`: 模型列表（所有leader或所有follower）
- `n_leaders` / `n_followers`: 数量信息（用于验证）

---

## 🔄 保存和加载流程

### 保存流程

```
训练循环
  ↓
每完成一个episode
  ↓
检查是否满足保存条件
  episode % save_interval == 0 AND episode > save_threshold
  ↓ 是
调用 save_models(output_dir)
  ↓
遍历所有actor
  ↓
保存每个actor的网络和优化器
  torch.save({'net': ..., 'opt': ...}, 'actor_i.pth')
```

### 加载流程

```
创建MASACTester
  ↓
调用 load_models(model_dir)
  ↓
遍历所有actor
  ↓
检查模型文件是否存在
  ↓ 存在
加载模型
  checkpoint = torch.load(path, map_location=device)
  actor.action_net.load_state_dict(checkpoint['net'])
```

---

## 🎯 自定义保存策略

### 1. 更频繁保存

```yaml
output:
  save_interval: 10     # 每10轮保存一次
  save_threshold: 100   # 100轮后开始保存
```

**效果**: 保存更频繁，但占用更多磁盘空间

### 2. 减少保存次数

```yaml
output:
  save_interval: 50     # 每50轮保存一次
  save_threshold: 300   # 300轮后开始保存
```

**效果**: 节省磁盘空间，但丢失中间状态

### 3. 从一开始就保存

```yaml
output:
  save_interval: 20
  save_threshold: 0     # 从第0轮开始
```

**效果**: 保存所有训练阶段的模型

---

## 📊 保存策略建议

### 推荐配置（不同场景）

#### 场景1: 正常训练（默认）

```yaml
output:
  save_interval: 20
  save_threshold: 200
```

**理由**:
- 前200轮模型质量较低，不需要保存
- 每20轮保存一次，平衡存储和安全性

#### 场景2: 调试模式

```yaml
output:
  save_interval: 10
  save_threshold: 0
```

**理由**:
- 频繁保存便于分析
- 从一开始就保存

#### 场景3: 长时间训练

```yaml
output:
  save_interval: 50
  save_threshold: 500
```

**理由**:
- 只保存后期收敛阶段的模型
- 节省磁盘空间

#### 场景4: 每轮都保存

```yaml
output:
  save_interval: 1
  save_threshold: 0
```

**理由**:
- 不错过任何一轮
- 用于详细分析训练过程

---

## 💾 磁盘空间估算

### 单个模型文件大小

**Actor网络** (hidden_dim=256):
- 参数量: ~200K
- 文件大小: ~1MB per actor

**示例**:
- 3个智能体: ~3MB per checkpoint
- 保存15次: ~45MB
- 500轮，每轮保存: ~1.5GB

---

## 🔧 代码实现

### save_models 方法

```python
def save_models(self, output_dir):
    """保存模型"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, actor in enumerate(self.actors):
        save_data = {
            'net': actor.action_net.state_dict(),  # 网络参数
            'opt': actor.optimizer.state_dict()    # 优化器状态
        }
        path = os.path.join(output_dir, f'actor_{i}.pth')
        torch.save(save_data, path)
    
    print(f"模型已保存到 {output_dir}")
```

### load_models 方法

```python
def load_models(self, output_dir):
    """加载模型"""
    for i, actor in enumerate(self.actors):
        path = os.path.join(output_dir, f'actor_{i}.pth')
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            actor.action_net.load_state_dict(checkpoint['net'])
            # 注意：测试时不加载优化器状态
```

---

## ⚠️ 注意事项

### 1. 只保存Actor网络

当前实现**只保存Actor**，不保存Critic和Entropy：

- ✅ Actor: 保存（用于测试）
- ❌ Critic: 不保存（仅训练时使用）
- ❌ Entropy: 不保存（仅训练时使用）

**原因**: 测试时只需要策略网络（Actor），不需要价值网络

### 2. 覆盖保存

每次保存会**覆盖**之前的文件：

```
第一次保存（Episode 220）: actor_0.pth, actor_1.pth, ...
第二次保存（Episode 240）: 覆盖 actor_0.pth, actor_1.pth, ...
```

**如果需要保留历史版本**，可以修改为：

```python
path = os.path.join(output_dir, f'actor_{i}_ep{episode}.pth')
```

### 3. GPU/CPU兼容

模型加载时支持设备转换：

```python
# GPU训练的模型可以在CPU上加载
checkpoint = torch.load(path, map_location=self.device)
```

---

## 🎓 使用示例

### 示例1: 使用默认保存设置

```bash
python scripts/masac/train.py --n_followers 3
```

**保存行为**:
- Episode 220, 240, 260, ..., 500: 保存模型

### 示例2: 自定义保存设置

```yaml
# 创建 configs/frequent_save.yaml
output:
  save_interval: 10     # 每10轮保存
  save_threshold: 50    # 50轮后开始
```

```bash
python scripts/masac/train.py --config configs/frequent_save.yaml
```

**保存行为**:
- Episode 60, 70, 80, ..., 500: 保存模型

### 示例3: 指定输出目录

```bash
python scripts/masac/train.py --n_followers 3 --output_dir output/exp1
```

**保存位置**: `output/exp1/actor_*.pth`

---

## 📝 总结

### 当前保存机制（已修正）

**配置参数** ✅:
```yaml
output:
  save_interval: 20     # 间隔
  save_threshold: 200   # 阈值
  output_dir: "output"  # 目录
```

**代码实现** ✅:
```python
if episode % self.save_interval == 0 and episode > self.save_threshold:
    self.save_models(output_dir)
```

**保存内容**:
- 每个Actor的网络参数
- 每个Actor的优化器状态

**保存格式**:
- PyTorch .pth 格式
- 文件名: `actor_{id}.pth`

**加载兼容**:
- ✅ GPU/CPU互通
- ✅ 自动设备映射

---

**已修正**: 配置参数现在正确使用！✨

