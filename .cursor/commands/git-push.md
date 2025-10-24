执行以下步骤将本地代码推送到 GitHub 远程仓库，使远程与本地保持一致：

**标准推送流程（推荐）：**

```bash
# 1. 查看当前状态
git status

# 2. 如果有未提交的修改，先添加并提交
git add .
git commit -m "<类型>: <简短描述>"

# 3. 拉取远程最新代码（避免冲突）
git pull origin main --rebase

# 4. 推送到远程仓库
git push origin main
```

**方案一：标准推送（有提交历史）**

适用于本地已有提交，需要推送到远程：

```bash
# 1. 确认当前分支
git branch

# 2. 查看本地提交
git log -3 --oneline

# 3. 推送到远程
git push origin main

# 或推送所有分支
git push origin --all
```

**方案二：首次推送（新仓库）**

适用于本地仓库首次推送到远程：

```bash
# 1. 添加远程仓库地址
git remote add origin <远程仓库URL>

# 2. 查看远程仓库
git remote -v

# 3. 首次推送并设置上游分支
git push -u origin main
```

**方案三：强制推送（谨慎使用）**

适用于需要用本地完全覆盖远程的情况：

```bash
# 警告：这会覆盖远程仓库，导致远程的修改丢失！

# 1. 查看本地和远程的差异
git fetch origin
git log origin/main..HEAD --oneline

# 2. 强制推送（仅在确认必要时使用）
git push origin main --force

# 或使用更安全的 force-with-lease（推荐）
git push origin main --force-with-lease
```

**方案四：推送标签**

推送代码的同时推送标签：

```bash
# 1. 推送代码
git push origin main

# 2. 推送所有标签
git push origin --tags

# 或推送单个标签
git push origin <标签名>
```

**快速执行命令（根据情况选择）：**

```bash
# 情况1: 标准推送（最常用）
git push origin main

# 情况2: 首次推送
git push -u origin main

# 情况3: 推送前先同步
git pull origin main --rebase && git push origin main

# 情况4: 推送所有分支和标签
git push origin --all && git push origin --tags
```

**完整工作流程：**

```bash
# 步骤1: 查看修改
git status

# 步骤2: 添加修改（遵守 .gitignore）
git add .

# 步骤3: 提交修改
git commit -m "feat: 完成某功能"

# 步骤4: 拉取最新代码
git pull origin main --rebase

# 步骤5: 推送到远程
git push origin main
```

**常见问题处理：**

1. **推送被拒绝（远程有新提交）**：
```bash
# 先拉取远程代码
git pull origin main --rebase

# 如果有冲突，解决冲突后
git add .
git rebase --continue

# 再次推送
git push origin main
```

2. **修改上一次提交后推送**：
```bash
# 修改上一次提交
git commit --amend -m "新的提交信息"

# 强制推送（仅限未被他人拉取的提交）
git push origin main --force-with-lease
```

3. **推送到不同分支**：
```bash
# 推送到指定分支
git push origin <分支名>

# 推送本地分支到远程不同名称的分支
git push origin <本地分支>:<远程分支>
```

4. **删除远程分支或标签**：
```bash
# 删除远程分支
git push origin --delete <分支名>

# 删除远程标签
git push origin --delete <标签名>
```

**检查推送结果：**

```bash
# 查看远程分支状态
git branch -r

# 查看本地和远程的差异
git fetch origin
git log origin/main..HEAD --oneline  # 本地领先的提交
git log HEAD..origin/main --oneline  # 远程领先的提交
```

**注意事项：**
- 推送前确保已提交所有本地修改
- 避免对 main/master 分支使用 `--force`，除非确实必要
- 使用 `--force-with-lease` 比 `--force` 更安全，会检查远程是否有新提交
- 推送前先拉取可以避免大多数冲突
- 团队协作时，不要强制推送已被他人拉取的提交
- 确保网络连接正常，大文件推送可能需要较长时间

