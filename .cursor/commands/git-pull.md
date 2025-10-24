执行以下步骤从 GitHub 仓库拉取最新代码，并保持本地与远程仓库完全一致：

**方案一：保留本地修改（推荐）**

如果本地有未提交的修改但想保留它们：

```bash
# 1. 查看当前状态
git status

# 2. 暂存本地修改
git stash

# 3. 拉取远程最新代码
git pull origin main

# 4. 恢复本地修改
git stash pop
```

**方案二：放弃本地修改（强制同步）**

如果本地有未提交的修改且不需要保留，强制与远程保持一致：

```bash
# 1. 查看当前状态
git status

# 2. 放弃所有本地修改
git reset --hard HEAD

# 3. 清理未跟踪的文件（可选）
git clean -fd

# 4. 拉取远程最新代码
git pull origin main

# 或者直接强制重置到远程分支
git fetch origin
git reset --hard origin/main
```

**方案三：标准拉取（无冲突时）**

如果本地没有修改或已经提交：

```bash
# 1. 查看当前状态
git status

# 2. 拉取远程最新代码
git pull origin main

# 或使用 rebase 方式保持提交历史整洁
git pull --rebase origin main
```

**快速执行命令（根据情况选择）：**

```bash
# 情况1: 保留本地修改
git stash && git pull origin main && git stash pop

# 情况2: 强制与远程一致（会丢失本地修改）
git fetch origin && git reset --hard origin/main && git clean -fd

# 情况3: 标准拉取
git pull origin main
```

**常见问题处理：**

1. **拉取时遇到冲突**：
```bash
# 如果使用 stash pop 后有冲突，手动解决冲突后
git add .
git stash drop
```

2. **切换到不同分支**：
```bash
# 查看所有分支
git branch -a

# 切换并拉取指定分支
git checkout <分支名>
git pull origin <分支名>
```

3. **查看拉取的更新内容**：
```bash
# 拉取前查看远程更新
git fetch origin
git log HEAD..origin/main --oneline

# 拉取后查看最新提交
git log -5 --oneline
```

**注意事项：**
- 使用 `git reset --hard` 会永久删除本地未提交的修改，请谨慎使用
- 建议在重要修改前先创建备份分支
- 如果不确定主分支名称，使用 `git branch -a` 查看
- 拉取前确保网络连接正常

