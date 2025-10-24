执行以下步骤进行 Git 提交：

1. 首先检查当前仓库状态，确保了解将要提交的更改
2. 如果项目中存在 .gitignore 文件，严格遵守其规则，不对 .gitignore 进行任何修改
3. 使用 `git add .` 添加所有符合 .gitignore 规则的文件
4. 使用规范的提交格式创建提交

**提交消息格式规范 (Conventional Commits)：**

```
<类型>: <简短描述>

<详细说明>(可选)
```

**常用类型：**
- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档修改
- `style`: 代码格式调整（不影响功能）
- `refactor`: 重构代码（既不是新功能也不是修复）
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动
- `build`: 构建系统或外部依赖变更
- `ci`: CI 配置文件和脚本的变更

**执行步骤：**

```bash
# 1. 查看当前状态
git status

# 2. 添加所有更改（遵守 .gitignore 规则）
git add .

# 3. 再次确认将要提交的内容
git status

# 4. 创建规范的提交
git commit -m "<类型>: <简短描述>"
# 或者使用多行提交信息
git commit -m "<类型>: <简短描述>" -m "<详细说明>"
```

**示例：**
- `git commit -m "feat: 添加 DDPG 算法实现"`
- `git commit -m "fix: 修复路径规划环境中的奖励计算错误"`
- `git commit -m "refactor: 优化 SAC 算法训练流程"`
- `git commit -m "docs: 更新 README 文档"`
- `git commit -m "perf: 提升环境渲染性能"`

**注意事项：**
- 提交消息使用中文或英文，保持项目一致性
- 简短描述不超过 50 个字符
- 详细说明可以包含更改的原因、影响等信息
- 严格遵守 .gitignore 文件的规则
- 不要修改 .gitignore 文件的内容

