# 训练输出文件说明

## 📁 输出文件完整列表

一次完整的训练会生成以下文件：

---

## 🗂️ 文件清单

### 1. 训练数据文件 (pkl格式)

**文件名**: `MASAC_new1`  
**格式**: Python pickle  
**位置**: `{output_dir}/MASAC_new1`  
**默认**: `output/MASAC_new1`

**内容**:
```python
{
    'all_ep_r_mean': np.array([episode_0_reward, episode_1_reward, ...])
}
```

**包含信息**:
- 每一轮(episode)的总奖励
- 数组形状: (max_episodes,)
- 用于后续分析和绘图

**生成时机**: 训练完成后

---

### 2. 训练曲线图 (PNG格式)

**文件名**: `training_curve.png`  
**格式**: PNG图片  
**位置**: `{output_dir}/training_curve.png`  
**默认**: `output/training_curve.png`

**内容**:
- X轴: Episode编号
- Y轴: 总奖励
- 红色曲线，标签"MASAC"
- 尺寸: 8×4英寸，DPI 150

**生成时机**: 训练完成后

---

### 3. Leader模型权重 (pth格式)

**文件名**: `leader.pth`  
**格式**: PyTorch模型文件  
**位置**: `{output_dir}/leader.pth`  
**默认**: `output/leader.pth`

**内容**:
```python
{
    'models': [
        {
            'net': leader_0_network_state_dict,  # 网络参数
            'opt': leader_0_optimizer_state_dict  # 优化器状态
        },
        # 如果有多个leader，继续...
    ],
    'n_leaders': 1  # Leader数量
}
```

**文件大小**: ~270-300 KB per leader  
**生成时机**: 
- 按配置的 `save_interval` 和 `save_threshold`
- 默认: 每20轮保存一次，200轮后开始
- 训练完成时最终保存

---

### 4. Follower模型权重 (pth格式)

**文件名**: `follower.pth`  
**格式**: PyTorch模型文件  
**位置**: `{output_dir}/follower.pth`  
**默认**: `output/follower.pth`

**内容**:
```python
{
    'models': [
        {
            'net': follower_0_network_state_dict,
            'opt': follower_0_optimizer_state_dict
        },
        {
            'net': follower_1_network_state_dict,
            'opt': follower_1_optimizer_state_dict
        },
        # 更多follower...
    ],
    'n_followers': 3  # Follower数量（例如3个）
}
```

**文件大小**: ~270-300 KB per follower  
**生成时机**: 
- 与leader.pth同时保存
- 每20轮保存一次（默认），200轮后开始

---

## 📂 完整目录结构

### 默认输出结构

```
UAV_PATH_PLANNING/
└── output/                          # 默认输出目录
    ├── MASAC_new1                   # 训练数据（无扩展名，pkl格式）
    ├── training_curve.png           # 训练曲线图
    ├── leader.pth                   # Leader模型权重
    └── follower.pth                 # Follower模型权重
```

### 自定义输出目录

如果使用 `--output_dir` 参数：

```bash
python scripts/masac/train.py --n_followers 3 --output_dir output/exp1
```

**生成结构**:
```
UAV_PATH_PLANNING/
└── output/
    └── exp1/                        # 自定义目录
        ├── MASAC_new1
        ├── training_curve.png
        ├── leader.pth
        └── follower.pth
```

---

## 🔧 配置控制

### 输出目录配置

**配置文件** (`configs/masac/default.yaml`):
```yaml
output:
  save_interval: 20       # 每隔多少轮保存模型
  save_threshold: 200     # 超过多少轮后才开始保存
  output_dir: "output"    # 输出目录
```

### 修改输出位置

**方式1: 修改配置文件**
```yaml
output:
  output_dir: "output/my_experiment"
```

**方式2: 命令行参数**
```bash
python scripts/masac/train.py --output_dir output/my_experiment
```

**方式3: 创建自定义配置**
```bash
cp configs/masac/default.yaml configs/my_config.yaml
# 编辑my_config.yaml中的output_dir
python scripts/masac/train.py --config configs/my_config.yaml
```

---

## 📊 文件大小估算

### 单次训练（默认配置：1 leader + 1 follower, 500轮）

| 文件 | 大小估算 | 说明 |
|-----|---------|------|
| `MASAC_new1` | ~4 KB | 500个float64数值 |
| `training_curve.png` | ~40-50 KB | PNG图片 |
| `leader.pth` | ~270 KB | 1个leader网络 |
| `follower.pth` | ~270 KB | 1个follower网络 |
| **总计** | **~580 KB** | 小于1 MB |

### 多follower训练（1 leader + 5 followers）

| 文件 | 大小估算 | 说明 |
|-----|---------|------|
| `MASAC_new1` | ~4 KB | 奖励数据 |
| `training_curve.png` | ~40-50 KB | 曲线图 |
| `leader.pth` | ~270 KB | 1个leader |
| `follower.pth` | **~1.35 MB** | 5个follower |
| **总计** | **~1.7 MB** | |

---

## 📝 文件用途说明

### 1. MASAC_new1 (训练数据)

**用途**:
- 后续分析训练过程
- 绘制更详细的图表
- 与其他实验对比

**如何使用**:
```python
import pickle as pkl

# 加载数据
with open('output/MASAC_new1', 'rb') as f:
    data = pkl.load(f)

rewards = data['all_ep_r_mean']
print(f"训练轮数: {len(rewards)}")
print(f"最佳奖励: {rewards.max()}")
print(f"最终奖励: {rewards[-1]}")

# 绘制自定义图表
import matplotlib.pyplot as plt
plt.plot(rewards)
plt.show()
```

---

### 2. training_curve.png (训练曲线)

**用途**:
- 直观查看训练进度
- 放入论文或报告
- 快速评估训练效果

**查看方式**:
```bash
# Linux
eog output/training_curve.png
# 或
display output/training_curve.png

# Windows
start output/training_curve.png

# Mac
open output/training_curve.png
```

---

### 3. leader.pth & follower.pth (模型权重)

**用途**:
- 测试训练好的策略
- 继续训练（恢复checkpoint）
- 部署到实际系统

**如何使用**:
```bash
# 测试模型
python scripts/masac/test.py \
    --n_followers 3 \
    --model_dir output/ \
    --render
```

**加载示例**:
```python
import torch

# 查看模型信息
leader_data = torch.load('output/leader.pth')
print(f"Leader数量: {leader_data['n_leaders']}")
print(f"包含模型: {len(leader_data['models'])}个")

follower_data = torch.load('output/follower.pth')
print(f"Follower数量: {follower_data['n_followers']}")
```

---

## 🕐 文件生成时间线

### 训练过程中的文件生成

```
训练开始
  ↓
Episode 0-200: 不保存模型（< save_threshold）
  ↓
Episode 220: ✅ 保存 leader.pth + follower.pth
  ↓
Episode 240: ✅ 保存 leader.pth + follower.pth（覆盖）
  ↓
Episode 260: ✅ 保存 leader.pth + follower.pth（覆盖）
  ↓
...
  ↓
Episode 500: ✅ 保存 leader.pth + follower.pth（覆盖）
  ↓
训练完成
  ↓
生成 MASAC_new1 ✅
生成 training_curve.png ✅
```

**注意**: 模型文件会被**覆盖**，只保留最新的版本

---

## 📋 实际示例

### 示例1: 默认配置训练

**命令**:
```bash
python scripts/masac/train.py
```

**生成文件**:
```
output/
├── MASAC_new1            # 500轮奖励数据
├── training_curve.png    # 训练曲线
├── leader.pth           # 最终模型（episode 500）
└── follower.pth         # 最终模型（episode 500）
```

---

### 示例2: 3个follower训练到自定义目录

**命令**:
```bash
python scripts/masac/train.py \
    --n_followers 3 \
    --output_dir output/3followers_exp1
```

**生成文件**:
```
output/
└── 3followers_exp1/
    ├── MASAC_new1
    ├── training_curve.png
    ├── leader.pth           # 1个leader
    └── follower.pth         # 3个follower，文件大小~810KB
```

---

### 示例3: 多个实验并行

**命令**:
```bash
# 实验1
python scripts/masac/train.py --n_followers 1 --output_dir output/exp1_1f &

# 实验2  
python scripts/masac/train.py --n_followers 3 --output_dir output/exp2_3f &

# 实验3
python scripts/masac/train.py --n_followers 5 --output_dir output/exp3_5f &
```

**生成结构**:
```
output/
├── exp1_1f/
│   ├── MASAC_new1
│   ├── training_curve.png
│   ├── leader.pth
│   └── follower.pth
├── exp2_3f/
│   ├── MASAC_new1
│   ├── training_curve.png
│   ├── leader.pth
│   └── follower.pth
└── exp3_5f/
    ├── MASAC_new1
    ├── training_curve.png
    ├── leader.pth
    └── follower.pth
```

---

## 🔄 文件使用流程

### 训练 → 测试流程

```bash
# 1. 训练
python scripts/masac/train.py --n_followers 3 --output_dir output/my_exp

# 生成文件:
# output/my_exp/MASAC_new1
# output/my_exp/training_curve.png
# output/my_exp/leader.pth
# output/my_exp/follower.pth

# 2. 查看训练曲线
display output/my_exp/training_curve.png

# 3. 测试模型
python scripts/masac/test.py \
    --n_followers 3 \
    --model_dir output/my_exp \
    --render

# 4. 分析数据
python -c "
import pickle as pkl
with open('output/my_exp/MASAC_new1', 'rb') as f:
    data = pkl.load(f)
    rewards = data['all_ep_r_mean']
    print(f'平均奖励: {rewards.mean():.2f}')
    print(f'最佳奖励: {rewards.max():.2f}')
"
```

---

## 📊 输出目录管理建议

### 1. 按实验组织

```
output/
├── baseline/           # 基线实验
├── exp_1follower/      # 1个follower
├── exp_3followers/     # 3个follower
├── exp_5followers/     # 5个follower
├── ablation_no_obs/    # 消融实验：无避障
└── compare_seeds/      # 不同种子对比
    ├── seed42/
    ├── seed123/
    └── seed456/
```

### 2. 按日期组织

```
output/
├── 20251024_exp1/
├── 20251024_exp2/
├── 20251025_exp1/
└── ...
```

### 3. 混合组织

```
output/
├── paper_experiments/       # 论文实验
│   ├── fig1_baseline/
│   ├── fig2_3followers/
│   └── fig3_5followers/
└── debug/                   # 调试测试
    └── quick_test/
```

---

## 💾 磁盘空间管理

### 空间估算

**单个实验** (~580 KB):
- 可以运行数千个实验

**100个实验** (~58 MB):
- 磁盘占用很小

**建议**:
- 定期清理debug实验
- 重要实验备份到单独目录
- 可以压缩历史实验

### 清理命令

```bash
# 清理所有输出
rm -rf output/*

# 只保留特定实验
rm -rf output/debug output/test*

# 压缩历史实验
tar -czf experiments_backup_20251024.tar.gz output/
```

---

## 🔍 快速检查

### 检查输出文件

```bash
# 查看输出目录
ls -lh output/

# 查看特定实验
ls -lh output/my_exp/

# 查看文件大小
du -sh output/*
```

### 验证文件完整性

```bash
# 检查必需文件是否存在
cd output/my_exp
ls MASAC_new1 training_curve.png leader.pth follower.pth

# 如果都存在，说明训练完整
```

---

## 📝 配置参考

### 当前默认配置

```yaml
# configs/masac/default.yaml
output:
  save_interval: 20       # 每20轮保存一次
  save_threshold: 200     # 200轮后开始保存
  output_dir: "output"    # 输出到output目录
```

### 保存时间点

**500轮训练的保存时间**:
- Episode 220 ✅
- Episode 240 ✅
- Episode 260 ✅
- Episode 280 ✅
- ...
- Episode 480 ✅
- Episode 500 ✅ (训练结束时)

**总共保存**: ~15次（会覆盖，最终只有1份）

---

## 🎯 常见场景

### 场景1: 快速测试

```yaml
# configs/quick.yaml
output:
  save_interval: 1
  save_threshold: 0
  output_dir: "output/quick_test"
  
training:
  max_episodes: 10
```

**输出**: `output/quick_test/` (每轮都保存)

---

### 场景2: 长时间训练

```yaml
# configs/long_train.yaml
output:
  save_interval: 50      # 减少保存频率
  save_threshold: 500
  output_dir: "output/long_train"

training:
  max_episodes: 2000
```

**输出**: `output/long_train/` (节省磁盘I/O)

---

### 场景3: 论文实验

```bash
# 有组织的输出
python scripts/masac/train.py \
    --n_followers 3 \
    --output_dir output/paper/fig2_3followers
```

**输出**: `output/paper/fig2_3followers/`

---

## 📚 相关命令

### 查看训练输出

```bash
# 查看所有实验
ls -R output/

# 查看特定实验的文件
ls -lh output/my_exp/

# 查看模型信息
python -c "
import torch
data = torch.load('output/leader.pth')
print(f'Leader数量: {data[\"n_leaders\"]}')
data = torch.load('output/follower.pth')  
print(f'Follower数量: {data[\"n_followers\"]}')
"
```

### 备份实验

```bash
# 备份重要实验
cp -r output/my_exp output/backup/my_exp_20251024

# 压缩备份
tar -czf my_exp.tar.gz output/my_exp/
```

---

## ⚠️ 注意事项

### 1. 文件覆盖

⚠️ **模型文件会被覆盖**
- 每次保存会覆盖之前的 `leader.pth` 和 `follower.pth`
- 如果需要保留历史版本，使用不同的output_dir

### 2. 路径相对性

**相对路径**: `output/my_exp`
- 相对于**项目根目录**
- 不是相对于当前工作目录

**绝对路径**: `/home/user/UAV_PATH_PLANNING/output/my_exp`
- 也可以使用绝对路径

### 3. 权限问题

确保有写入权限:
```bash
chmod +w output/
```

---

## ✅ 总结

### 每次训练输出文件

**固定4个文件**:
1. `MASAC_new1` - 训练数据（pkl）
2. `training_curve.png` - 训练曲线（PNG）
3. `leader.pth` - Leader模型权重
4. `follower.pth` - Follower模型权重

**默认位置**: `output/`

**自定义**: 通过 `--output_dir` 参数或配置文件

**总大小**: ~580 KB (1 leader + 1 follower)  
          ~1-2 MB (1 leader + 多个follower)

---

**文档版本**: v1.0  
**更新日期**: 2025-10-25

