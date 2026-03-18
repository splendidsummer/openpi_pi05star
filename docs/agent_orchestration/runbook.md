# 多智能体编排 Runbook（可执行草案）

## 1. 启用 Agent Teams（实验特性）

在 Claude Code 设置中加入：

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  },
  "teammateMode": "in-process"
}
```

## 2. 载入编排草案

- 主编排：`docs/agent_orchestration/team_config.yaml`
- 任务模板：`docs/agent_orchestration/task_templates.yaml`

建议在会话开始时把这两份文件作为上下文输入给 lead agent（orchestrator）。

## 3. 启动顺序（建议）

1) `phase_0_bootstrap`
- 启动 `orchestrator` 与 `debug_agent`

2) `phase_1_specification`
- 并行启动 `model_architect`、`data_explorer`
- 检查 `gate_A_spec_ready`

3) `phase_2_foundation_build`
- 并行启动 `data_loader`、`normalizer`、`training_dev`、`inference_dev`

4) `phase_3_preflight_validation`
- 并行启动 `module_tester`、`safety_validator`、`integration_tester`
- 检查 `gate_C_release_ready`

5) `phase_4_execution_branches`
- 并行分支：`training_runner` / `evaluation_agent` / `inference_deployer`

6) `phase_5_convergence`
- `orchestrator` 汇总发布决策
- `debug_agent` 生成回流任务（若不达标）

## 4. 回流策略（失败处理）

- 单任务失败：先重试最多2次，仍失败则交由 `debug_agent` 分类。
- 分类完成后，`orchestrator` 只重跑受影响 phase，不重置全链路。
- 严格要求每个任务输出产物契约（artifact + metrics + next_actions）。

## 5. OpenPI 推荐命令（按需）

```bash
# 训练（JAX）
uv run python scripts/train.py --config-name=pi05_droid

# 训练（PyTorch）
uv run python scripts/train_pytorch.py --config-name=pi05_droid

# 推理服务
uv run python scripts/serve_policy.py --port=8000

# 关键测试
pytest --strict-markers src/openpi/models/ -m "not manual"
pytest --strict-markers src/openpi/transforms_test.py -m "not manual"
```

## 6. 落地建议

- 将 `.claude/artifacts` 作为团队共享状态目录。
- 在 PR 模板中要求附带对应 phase 的 artifact 路径。
- 对 GPU 任务设置并发上限，避免训练与评估同时抢占导致抖动。
