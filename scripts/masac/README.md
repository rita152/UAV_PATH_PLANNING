# MASAC训练和测试脚本

本目录包含独立的训练和测试脚本，从 `main_SAC.py` 重构而来，功能保持一致。

## 📁 文件说明

- `train.py` - 训练脚本
- `test.py` - 测试脚本
- `__init__.py` - 包初始化文件
- `README.md` - 本文档

---

## 🚀 使用方法

### 训练模式 (train.py)

#### 基本使用

```bash
# 使用默认配置训练
python scripts/masac/train.py

# 或使用conda环境
conda run -n UAV_PATH_PLANNING python scripts/masac/train.py
```

#### 高级用法

```bash
# 指定follower数量
python scripts/masac/train.py --n_followers 3

# 使用自定义配置文件
python scripts/masac/train.py --config configs/my_config.yaml

# 开启可视化渲染（会降低训练速度）
python scripts/masac/train.py --render

# 指定输出目录
python scripts/masac/train.py --output_dir output/exp1

# 组合参数
python scripts/masac/train.py --n_followers 3 --config configs/masac/default.yaml --output_dir output/3followers
```

#### 命令行参数

```
--config CONFIG           配置文件路径 (默认: configs/masac/default.yaml)
--n_followers N_FOLLOWERS 跟随者数量（覆盖配置文件）
--render                  是否开启可视化渲染
--output_dir OUTPUT_DIR   输出目录（覆盖配置文件）
```

---

### 测试模式 (test.py)

#### 基本使用

```bash
# 使用默认配置测试
python scripts/masac/test.py

# 或使用conda环境
conda run -n UAV_PATH_PLANNING python scripts/masac/test.py
```

#### 高级用法

```bash
# 指定follower数量
python scripts/masac/test.py --n_followers 3

# 开启可视化渲染（推荐）
python scripts/masac/test.py --render

# 指定模型目录
python scripts/masac/test.py --model_dir output/exp1

# 指定测试轮数
python scripts/masac/test.py --test_episodes 50

# 组合参数
python scripts/masac/test.py --n_followers 3 --render --model_dir output/3followers --test_episodes 100
```

#### 命令行参数

```
--config CONFIG           配置文件路径 (默认: configs/masac/default.yaml)
--n_followers N_FOLLOWERS 跟随者数量（覆盖配置文件）
--render                  是否开启可视化渲染
--model_dir MODEL_DIR     模型目录路径（默认使用配置文件中的output_dir）
--test_episodes EPISODES  测试轮数（覆盖配置文件）
```

---

## 📊 与 main_SAC.py 的对比

### main_SAC.py (原版)

```bash
# 训练
python main_SAC.py --n_followers 3

# 测试
python main_SAC.py --test --n_followers 3 --render
```

### train.py / test.py (新版)

```bash
# 训练
python scripts/masac/train.py --n_followers 3

# 测试
python scripts/masac/test.py --n_followers 3 --render
```

---

## ✨ 改进特性

### 相比 main_SAC.py 的优势

1. **职责单一**
   - train.py 专注于训练
   - test.py 专注于测试
   - 代码更清晰，更易维护

2. **更好的代码组织**
   - 函数模块化
   - 每个函数职责明确
   - 便于理解和修改

3. **独立运行**
   - 训练和测试完全分离
   - 互不影响
   - 可以在不同机器上运行

4. **保存训练曲线**
   - train.py 会自动保存训练曲线图
   - 保存为 PNG 文件便于查看

5. **额外的参数**
   - train.py: --output_dir（自定义输出目录）
   - test.py: --model_dir（指定模型目录）
   - test.py: --test_episodes（自定义测试轮数）

---

## 🎯 使用示例

### 示例1: 完整训练流程

```bash
# 1. 训练1个follower
python scripts/masac/train.py --n_followers 1 --output_dir output/1follower

# 2. 训练3个follower
python scripts/masac/train.py --n_followers 3 --output_dir output/3followers

# 3. 训练5个follower
python scripts/masac/train.py --n_followers 5 --output_dir output/5followers
```

### 示例2: 测试已训练模型

```bash
# 测试1个follower模型（带可视化）
python scripts/masac/test.py --n_followers 1 --model_dir output/1follower --render

# 测试3个follower模型
python scripts/masac/test.py --n_followers 3 --model_dir output/3followers --render

# 测试5个follower模型（50轮测试）
python scripts/masac/test.py --n_followers 5 --model_dir output/5followers --test_episodes 50 --render
```

### 示例3: 使用不同配置文件

```bash
# 创建自定义配置
cp configs/masac/default.yaml configs/exp1.yaml
# 编辑 exp1.yaml 修改参数

# 使用自定义配置训练
python scripts/masac/train.py --config configs/exp1.yaml --n_followers 3
```

---

## 📁 输出文件

### train.py 输出

```
output/
├── MASAC_new1              # 训练数据（pickle格式）
├── training_curve.png      # 训练曲线图
├── actor_0.pth             # Agent 0的模型
├── actor_1.pth             # Agent 1的模型
└── ...
```

### 文件说明

- `MASAC_new1`: 包含所有episode的奖励数据
- `training_curve.png`: 训练曲线可视化
- `actor_*.pth`: 每个智能体的策略网络

---

## 🔧 故障排除

### 问题1: ModuleNotFoundError

**错误**: `ModuleNotFoundError: No module named 'xxx'`

**解决方案**:
```bash
# 安装依赖
pip install -r requirements.txt

# 或使用conda环境
conda run -n UAV_PATH_PLANNING python scripts/masac/train.py
```

### 问题2: 找不到配置文件

**错误**: `配置文件不存在: configs/masac/default.yaml`

**解决方案**:
```bash
# 确保在项目根目录运行
cd /path/to/UAV_PATH_PLANNING
python scripts/masac/train.py
```

### 问题3: 模型文件不存在

**错误**: 测试时找不到模型

**解决方案**:
```bash
# 确保先进行训练
python scripts/masac/train.py --n_followers 3

# 然后使用相同的配置测试
python scripts/masac/test.py --n_followers 3
```

---

## 🎓 最佳实践

### 1. 训练实验管理

为不同实验使用不同的输出目录：

```bash
# 实验1: 1个follower
python scripts/masac/train.py --n_followers 1 --output_dir output/exp1_1follower

# 实验2: 3个follower，更长训练
python scripts/masac/train.py --n_followers 3 --output_dir output/exp2_3followers

# 实验3: 使用不同种子
# 修改配置文件的seed -> base_seed
python scripts/masac/train.py --config configs/seed123.yaml --output_dir output/exp3_seed123
```

### 2. 批量测试

创建测试脚本 `batch_test.sh`:

```bash
#!/bin/bash
# 批量测试不同配置

echo "测试1个follower"
python scripts/masac/test.py --n_followers 1 --model_dir output/1follower --test_episodes 100

echo "测试3个follower"
python scripts/masac/test.py --n_followers 3 --model_dir output/3followers --test_episodes 100

echo "测试5个follower"
python scripts/masac/test.py --n_followers 5 --model_dir output/5followers --test_episodes 100
```

### 3. 可视化测试

```bash
# 测试时开启渲染查看效果
python scripts/masac/test.py --n_followers 3 --render --test_episodes 10
```

---

## 📚 相关文档

- `../../configs/masac/default.yaml` - 默认配置文件
- `../../main_SAC.py` - 原始脚本（仍可使用）
- `../../algorithm/masac/` - MASAC算法实现
- `../../utils/` - 工具模块（配置加载、种子管理等）

---

## 🔄 向后兼容

原来的 `main_SAC.py` 仍然可用，您可以继续使用：

```bash
# 训练
python main_SAC.py --n_followers 3

# 测试
python main_SAC.py --test --n_followers 3 --render
```

新的 `train.py` 和 `test.py` 提供了更好的代码组织，推荐使用。

---

**更新日期**: 2025-10-24  
**版本**: v1.0

