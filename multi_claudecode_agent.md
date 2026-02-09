多 Agent 协作指南：基于 Git Worktree 的并行开发流

在 2026 年的开发环境下，使用多个 AI Agent（如 Claude Code, GitHub Copilot CLI 等）协作时，最有效的隔离方式是使用 git worktree。本指南将帮助你建立一套互不干扰的并行开发工作流。

1. 核心理念：物理隔离

为什么要用 Worktree？

避免冲突：防止多个 Agent 同时写入同一个 index.lock 或修改同一份代码。

上下文纯净：每个 Agent 只能看到自己分支下的文件变化，不会被其他 Agent 的中间产物干扰。

极速切换：无需频繁 git stash 或 git checkout，多个工作区同时并存。

2. 标准操作流程 (SOP)

第一步：初始化环境

确保你在主仓库目录，且状态是干净的。

cd your-project-main
git checkout main
git pull origin main


第二步：创建 Agent 工作区

为每个 Agent 任务创建一个独立的工作路径和分支。

# 指令格式：git worktree add ../<工作区名称> -b <新分支名>

# 任务 A：UI 优化
git worktree add ../agent-ui-task -b feat/ui-refresh

# 任务 B：API 重构
git worktree add ../agent-api-task -b refactor/api-v2


第三步：启动 Agent 协作

在不同的终端窗口中分别进入对应的目录并启动 Agent。

终端 1 (Agent A):

cd ../agent-ui-task
# 启动你的 Agent (例如 Claude Code)
claude
# 任务指令："修改侧边栏布局，使其适配 4K 屏幕。"


终端 2 (Agent B):

cd ../agent-api-task
# 启动你的 Agent
claude
# 任务指令："将所有的 Axios 调用替换为原生 fetch，并添加错误捕获。"


3. 合并与收尾 (Post-Task)

当 Agent 完成任务并执行了 git commit 后，你需要将成果合并回主干。

返回主仓库:

cd ../your-project-main


合并分支:

git merge feat/ui-refresh
git merge refactor/api-v2


移除工作区:

# 移除物理目录及 worktree 记录
git worktree remove ../agent-ui-task
git worktree remove ../agent-api-task

# 删除已合并的本地分支
git branch -d feat/ui-refresh
git branch -d refactor/api-v2


4. 关键注意事项 💡

1. 环境变量 (.env)

由于 .env 文件通常在 .gitignore 中，新创建的 worktree 目录里不会有这些文件。

对策：创建 worktree 后，手动拷贝环境文件：cp .env ../agent-ui-task/。

2. 依赖管理 (node_modules)

硬链接/软链接：如果项目很大，建议在父级目录安装依赖，或者确保 Agent 知道在新的 worktree 目录里也需要运行一次 npm install (或者使用 pnpm 这种共享存储的工具)。

3. Git 状态锁定

Git 不允许在两个不同的 worktree 中同时 checkout 同一个分支。这是一种保护机制，确保 Agent 不会互相覆盖。

5. 快速速查表

命令

作用

git worktree list

查看当前所有的工作区

git worktree add <path> -b <branch>

创建新工作区及分支

git worktree remove <path>

移除指定工作区

git worktree prune

清理已手动删除目录的残留记录
