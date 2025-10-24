# main_SAC.py vs train.py/test.py 功能对比

## 📊 详细对比

### 训练功能对比 (main_SAC.py vs train.py)

| 功能点 | main_SAC.py | train.py | 是否一致 |
|-------|------------|----------|---------|
| **配置加载** | ✅ ConfigLoader | ✅ ConfigLoader | ✅ 一致 |
| **命令行参数** | --config, --n_followers, --render | --config, --n_followers, --render, --output_dir | ⚠️ train.py多了--output_dir |
| **随机种子** | ✅ setup_seeds | ✅ setup_seeds | ✅ 一致 |
| **环境创建** | ✅ RlGame | ✅ RlGame | ✅ 一致 |
| **训练器** | ✅ MASACTrainer | ✅ MASACTrainer | ✅ 一致 |
| **训练执行** | ✅ trainer.train() | ✅ trainer.train() | ✅ 一致 |
| **数据保存** | ✅ pkl.dump到shoplistfile | ✅ pkl.dump到shoplistfile | ✅ 一致 |
| **绘制曲线** | ✅ plt.plot | ✅ plt.plot | ✅ 一致 |
| **显示/保存图表** | ❗ plt.show() | ❗ plt.savefig() + 注释了show | ⚠️ **不一致** |
| **输出信息** | 平均奖励 | 平均奖励 + 最佳奖励 | ⚠️ train.py多了最佳奖励 |
| **环境关闭** | ✅ env.close() | ✅ env.close() | ✅ 一致 |

### 测试功能对比 (main_SAC.py vs test.py)

| 功能点 | main_SAC.py | test.py | 是否一致 |
|-------|------------|----------|---------|
| **配置加载** | ✅ ConfigLoader | ✅ ConfigLoader | ✅ 一致 |
| **命令行参数** | --config, --n_followers, --render, --test | --config, --n_followers, --render, --model_dir, --test_episodes | ⚠️ test.py多了参数 |
| **随机种子** | ✅ setup_seeds | ✅ setup_seeds | ✅ 一致 |
| **环境创建** | ✅ RlGame | ✅ RlGame | ✅ 一致 |
| **测试器** | ✅ MASACTester | ✅ MASACTester | ✅ 一致 |
| **模型加载** | ✅ load_models(OUTPUT_DIR) | ✅ load_models(model_dir) | ✅ 一致 |
| **测试执行** | ✅ tester.test(render) | ✅ tester.test(render) | ✅ 一致 |
| **环境关闭** | ✅ env.close() | ✅ env.close() | ✅ 一致 |

---

## ⚠️ 发现的差异

### 差异1: 训练曲线显示 vs 保存

**main_SAC.py (第159行)**:
```python
plt.show()  # 显示图表窗口，阻塞程序
```

**train.py (第210-214行)**:
```python
plt.savefig(os.path.join(config['output_dir'], 'training_curve.png'))
print(f"训练曲线已保存到: {config['output_dir']}/training_curve.png")

# 可选：显示图表
# plt.show()
```

**影响**: 
- ❌ **不一致**: 老版本会弹出图表窗口，新版本保存为文件
- ⚠️ 新版本需要修正以保持一致

### 差异2: 额外的参数

**train.py**:
- 新增 `--output_dir` 参数

**test.py**:
- 新增 `--model_dir` 参数
- 新增 `--test_episodes` 参数

**影响**:
- ✅ **向前兼容**: 新增功能，不影响原有使用方式
- ✅ 这些是增强功能，可以保留

### 差异3: 输出信息

**train.py 多了**:
```python
print(f"最佳奖励: {np.max(all_rewards):.2f}")
print(f"训练数据已保存到: {shoplistfile}")
print(f"训练曲线已保存到: ...")
```

**影响**:
- ✅ **向前兼容**: 增强的输出信息，有益无害

---

## 🔧 需要修正的问题

### 问题: plt.show() vs plt.savefig()

**建议修正方案**:
```python
# train.py 中应该保持与原版一致
plt.savefig(os.path.join(config['output_dir'], 'training_curve.png'))
print(f"训练曲线已保存到: {config['output_dir']}/training_curve.png")
plt.show()  # ← 添加这一行，与原版保持一致
```

这样既保存文件（新功能），又显示窗口（保持兼容）。

