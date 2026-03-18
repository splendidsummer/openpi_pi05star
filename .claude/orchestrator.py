#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception as exc:  # pragma: no cover
    print("缺少依赖: pyyaml，请先安装: uv pip install pyyaml", file=sys.stderr)
    raise SystemExit(2) from exc


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "docs" / "agent_orchestration" / "team_config.yaml"


@dataclass
class Task:
    task_id: str
    agent: str
    action: str
    needs: list[str]
    produces: list[str]


def agent_config_to_cli_name(agent_name: str) -> str:
    return agent_name.replace("_", "-")


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"未找到配置文件: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("team_config.yaml 顶层必须是对象")
    return data


def flatten_tasks(phase: dict[str, Any]) -> list[Task]:
    mode = phase.get("mode")
    items: list[dict[str, Any]] = []
    if mode == "parallel":
        items = phase.get("tasks", []) or []
    elif mode == "interactive":
        items = phase.get("tasks", []) or []
    elif mode == "parallel_branches":
        branches = phase.get("branches", {}) or {}
        for _, tasks in branches.items():
            items.extend(tasks or [])
    else:
        raise ValueError(f"不支持的 phase mode: {mode}")

    out: list[Task] = []
    for t in items:
        out.append(
            Task(
                task_id=t.get("id", ""),
                agent=t.get("agent", ""),
                action=t.get("action", ""),
                needs=t.get("needs", []) or [],
                produces=t.get("produces", []) or [],
            )
        )
    return out


def get_phase(cfg: dict[str, Any], phase_id: str) -> dict[str, Any]:
    phases = cfg.get("phases", []) or []
    for phase in phases:
        if phase.get("id") == phase_id:
            return phase
    raise ValueError(f"未找到 phase: {phase_id}")


def build_task_map(tasks: list[Task]) -> dict[str, Task]:
    return {task.task_id: task for task in tasks}


def local_needs(task: Task, task_map: dict[str, Task]) -> list[str]:
    return [dep for dep in task.needs if dep in task_map]


def external_needs(task: Task, task_map: dict[str, Task]) -> list[str]:
    return [dep for dep in task.needs if dep not in task_map]


def topo_layers(tasks: list[Task]) -> list[list[Task]]:
    task_map = build_task_map(tasks)

    indegree: dict[str, int] = {task.task_id: 0 for task in tasks}
    children: dict[str, list[str]] = {task.task_id: [] for task in tasks}

    for task in tasks:
        for dep in local_needs(task, task_map):
            indegree[task.task_id] += 1
            children[dep].append(task.task_id)

    ready = sorted([task_id for task_id, degree in indegree.items() if degree == 0])
    layers: list[list[Task]] = []
    visited = 0

    while ready:
        current_layer_ids = ready
        ready = []
        layer_tasks = [task_map[task_id] for task_id in current_layer_ids]
        layers.append(layer_tasks)
        visited += len(current_layer_ids)

        for parent in current_layer_ids:
            for child in sorted(children[parent]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        ready = sorted(set(ready))

    if visited != len(tasks):
        cycle_nodes = [task_id for task_id, degree in indegree.items() if degree > 0]
        raise ValueError(f"检测到循环依赖: {','.join(sorted(cycle_nodes))}")

    return layers


def task_completed(task: Task) -> bool:
    if not task.produces:
        return False
    return all((ROOT / artifact).exists() for artifact in task.produces)


def phase_status(cfg: dict[str, Any], phase_id: str) -> int:
    phase = get_phase(cfg, phase_id)
    tasks = flatten_tasks(phase)

    completed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for task in tasks:
        item = {
            "id": task.task_id,
            "agent": task.agent,
            "needs": task.needs,
            "produces": task.produces,
        }
        if task_completed(task):
            completed.append(item)
        else:
            pending.append(item)

    print(
        json.dumps(
            {
                "phase_id": phase_id,
                "mode": phase.get("mode"),
                "completed_count": len(completed),
                "pending_count": len(pending),
                "completed": completed,
                "pending": pending,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def next_tasks(cfg: dict[str, Any], phase_id: str, completed_ids: list[str]) -> int:
    phase = get_phase(cfg, phase_id)
    tasks = flatten_tasks(phase)
    task_map = build_task_map(tasks)

    completed_set = set(filter(None, completed_ids))
    completed_set.update({task.task_id for task in tasks if task_completed(task)})

    runnable: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    for task in tasks:
        if task.task_id in completed_set:
            continue
        unmet = [dep for dep in task.needs if dep not in completed_set]
        external_unmet = [dep for dep in external_needs(task, task_map) if dep not in completed_set]
        local_unmet = [dep for dep in local_needs(task, task_map) if dep not in completed_set]
        item = {
            "id": task.task_id,
            "agent": task.agent,
            "agent_cli_name": agent_config_to_cli_name(task.agent),
            "action": task.action,
            "needs": task.needs,
            "produces": task.produces,
        }
        if unmet:
            item["blocked_by"] = unmet
            if local_unmet:
                item["blocked_by_local"] = local_unmet
            if external_unmet:
                item["blocked_by_external"] = external_unmet
            blocked.append(item)
        else:
            runnable.append(item)

    print(
        json.dumps(
            {
                "phase_id": phase_id,
                "runnable": runnable,
                "blocked": blocked,
                "completed_ids": sorted(completed_set),
                "known_tasks": sorted(task_map.keys()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def export_phase_dispatch(cfg: dict[str, Any], phase_id: str) -> int:
    phase = get_phase(cfg, phase_id)
    mode = phase.get("mode")
    tasks = flatten_tasks(phase)
    layers = topo_layers(tasks)

    layer_payload: list[dict[str, Any]] = []
    for i, layer in enumerate(layers, start=1):
        payload_tasks = []
        for task in layer:
            payload_tasks.append(
                {
                    "id": task.task_id,
                    "agent": task.agent,
                    "agent_cli_name": agent_config_to_cli_name(task.agent),
                    "action": task.action,
                    "needs": task.needs,
                    "produces": task.produces,
                    "dispatch_prompt": (
                        f"执行任务 {task.task_id}: {task.action}。"
                        f"读取 docs/agent_orchestration/task_templates.yaml 中 {task.agent} 模板，"
                        "仅输出契约要求的 JSON，并写入产物路径。"
                    ),
                }
            )
        layer_payload.append({"layer": i, "tasks": payload_tasks})

    output = {
        "phase_id": phase_id,
        "mode": mode,
        "gate_on_complete": phase.get("gate_on_complete"),
        "layers": layer_payload,
    }

    out = ROOT / ".claude" / "artifacts" / "spec" / f"dispatch_{phase_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成 dispatch 包: {out.relative_to(ROOT)}")
    return 0


def inspect_artifact(rel_path: str) -> dict[str, Any]:
    path = ROOT / rel_path
    info: dict[str, Any] = {
        "artifact": rel_path,
        "exists": path.exists(),
        "absolute_path": str(path),
    }
    if not path.exists():
        return info

    stat = path.stat()
    info.update(
        {
            "is_file": path.is_file(),
            "size_bytes": stat.st_size,
            "modified_at": dt.datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }
    )

    if path.suffix.lower() == ".json" and path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            info["json_valid"] = True
            info["json_type"] = type(payload).__name__
        except Exception as exc:
            info["json_valid"] = False
            info["json_error"] = str(exc)

    return info


def validate_gate_rules(gate_id: str, gate: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    if gate_id != "gate_0_5_config_ready":
        return checks

    config_rel = ".claude/artifacts/config/data_paths.json"
    config_path = ROOT / config_rel
    if not config_path.exists():
        checks.append(
            {
                "name": "data_paths_exists",
                "passed": False,
                "message": "data_paths.json 不存在",
            }
        )
        return checks

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        checks.append(
            {
                "name": "data_paths_json_valid",
                "passed": False,
                "message": f"data_paths.json 不是合法 JSON: {exc}",
            }
        )
        return checks

    required_fields = [
        "raw_dataset_path",
        "processed_dataset_path",
        "norm_stats_path",
    ]
    missing_fields = [field for field in required_fields if not payload.get(field)]
    checks.append(
        {
            "name": "required_fields_present",
            "passed": len(missing_fields) == 0,
            "message": "所有必需字段已填写" if not missing_fields else f"缺少字段: {', '.join(missing_fields)}",
        }
    )

    raw_dataset_path = payload.get("raw_dataset_path", "")
    raw_path = Path(raw_dataset_path) if raw_dataset_path else None
    raw_path_exists = bool(raw_path and raw_path.exists() and raw_path.is_dir())
    checks.append(
        {
            "name": "raw_dataset_path_exists",
            "passed": raw_path_exists,
            "message": "raw_dataset_path 路径存在" if raw_path_exists else "raw_dataset_path 无效或不存在",
        }
    )

    parquet_ok = False
    if raw_path_exists and raw_path is not None:
        parquet_ok = any(raw_path.glob("*.parquet"))
    checks.append(
        {
            "name": "raw_dataset_has_parquet",
            "passed": parquet_ok,
            "message": "raw_dataset_path 包含 parquet 文件" if parquet_ok else "raw_dataset_path 中未找到 parquet 文件",
        }
    )

    for optional_dir_field in ["processed_dataset_path", "norm_stats_path", "checkpoint_path", "evaluation_output_path"]:
        value = payload.get(optional_dir_field)
        if not value:
            continue
        candidate = Path(value)
        checks.append(
            {
                "name": f"dir_exists::{optional_dir_field}",
                "passed": candidate.exists(),
                "message": f"{optional_dir_field} 已存在" if candidate.exists() else f"{optional_dir_field} 不存在",
            }
        )

    return checks


def check_gate(cfg: dict[str, Any], gate_id: str, details: bool = False, as_json: bool = False) -> int:
    gates = cfg.get("gates", {}) or {}
    gate = gates.get(gate_id)
    if gate is None:
        print(f"未知 gate: {gate_id}", file=sys.stderr)
        return 2

    required = gate.get("requires_artifacts", []) or []
    artifact_details = [inspect_artifact(rel) for rel in required]
    missing = [item["artifact"] for item in artifact_details if not item.get("exists")]
    rule_checks = validate_gate_rules(gate_id, gate)
    failed_rules = [check for check in rule_checks if not check.get("passed")]

    passed = (len(missing) == 0) and (len(failed_rules) == 0)

    payload = {
        "gate_id": gate_id,
        "description": gate.get("description"),
        "required_count": len(required),
        "missing_count": len(missing),
        "missing_artifacts": missing,
        "rule_failed_count": len(failed_rules),
        "rules": rule_checks,
        "artifacts": artifact_details,
        "passed": passed,
    }

    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if passed else 1

    if passed:
        print(f"[PASS] {gate_id} 校验通过")
    else:
        print(f"[FAIL] {gate_id} 校验失败")

    print(f"- 描述: {gate.get('description', '')}")
    print(f"- 产物: {len(required)} 项，缺失 {len(missing)} 项")
    if missing:
        for artifact in missing:
            print(f"  · 缺失: {artifact}")

    if rule_checks:
        print(f"- 规则校验: {len(rule_checks)} 项，失败 {len(failed_rules)} 项")
        for check in rule_checks:
            mark = "✓" if check.get("passed") else "✗"
            print(f"  {mark} {check.get('name')}: {check.get('message')}")

    if details:
        print("- 产物详情:")
        for item in artifact_details:
            if not item.get("exists"):
                print(f"  · {item['artifact']}: 不存在")
                continue
            extra = f"大小={item.get('size_bytes', 0)}B, 修改时间={item.get('modified_at', '-') }"
            if "json_valid" in item:
                extra += f", JSON合法={item.get('json_valid')}"
            print(f"  · {item['artifact']}: 存在, {extra}")

    return 0 if passed else 1


def show_plan(cfg: dict[str, Any]) -> int:
    phases = cfg.get("phases", []) or []
    print(f"project: {cfg.get('project', {}).get('name', 'unknown')}")
    print(f"phases: {len(phases)}")

    for p in phases:
        phase_id = p.get("id", "unknown_phase")
        mode = p.get("mode", "unknown_mode")
        gate = p.get("gate_on_complete")
        tasks = flatten_tasks(p)

        print(f"\n- {phase_id} [{mode}] tasks={len(tasks)}")
        if gate:
            print(f"  gate: {gate}")
        for t in tasks:
            dep = f" needs={','.join(t.needs)}" if t.needs else ""
            print(f"  * {t.task_id} | {t.agent} | {t.action}{dep}")
    return 0


def dry_run(cfg: dict[str, Any]) -> int:
    phases = cfg.get("phases", []) or []
    artifact_root = ROOT / ".claude" / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    state = {
        "project": cfg.get("project", {}).get("name", "unknown"),
        "phases": [],
    }

    for p in phases:
        phase_id = p.get("id", "unknown_phase")
        gate = p.get("gate_on_complete")
        tasks = flatten_tasks(p)
        task_state = []

        for t in tasks:
            raw_task_map = {
                "id": t.task_id,
                "agent": t.agent,
                "status": "pending",
                "needs": t.needs,
                "produces": t.produces,
            }
            mode = p.get("mode")
            if mode == "interactive":
                source_tasks = p.get("tasks", []) or []
                matched = next((x for x in source_tasks if x.get("id") == t.task_id), {})
                prompts = matched.get("interactive_prompts", []) or []
                raw_task_map["interactive_prompts"] = prompts

            task_state.append(
                raw_task_map
            )

        state["phases"].append({"id": phase_id, "tasks": task_state, "gate": gate})

    out = artifact_root / "spec" / "dry_run_plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成 dry-run 计划: {out.relative_to(ROOT)}")
    return 0


def ensure_artifact_tree() -> int:
    dirs = [
        ".claude/artifacts/spec",
        ".claude/artifacts/config",
        ".claude/artifacts/build",
        ".claude/artifacts/run/checkpoints",
        ".claude/artifacts/run/evaluation",
        ".claude/artifacts/run/deploy",
    ]
    for d in dirs:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
        print(f"ok: {d}")
    return 0


def _read_value(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    if value:
        return value
    return default or ""


def collect_user_config(args: argparse.Namespace) -> int:
    raw_dataset_path = args.raw_dataset_path
    processed_dataset_path = args.processed_dataset_path
    norm_stats_path = args.norm_stats_path
    checkpoint_path = args.checkpoint_path
    evaluation_output_path = args.evaluation_output_path

    if not raw_dataset_path:
        raw_dataset_path = _read_value("请指定原始数据集输入路径")
    if not processed_dataset_path:
        processed_dataset_path = _read_value("请指定处理后数据输出路径")
    if not norm_stats_path:
        norm_stats_path = _read_value("请指定归一化统计输出路径")

    if not checkpoint_path:
        checkpoint_path = _read_value("请指定 checkpoint 输出路径（可选）", ".claude/artifacts/run/checkpoints")
    if not evaluation_output_path:
        evaluation_output_path = _read_value("请指定评估输出路径（可选）", ".claude/artifacts/run/evaluation")

    raw_path = Path(raw_dataset_path)
    if not raw_path.exists() or not raw_path.is_dir():
        print(f"raw_dataset_path 无效: {raw_dataset_path}", file=sys.stderr)
        return 1
    if not any(raw_path.glob("*.parquet")):
        print("raw_dataset_path 目录中未找到 parquet 文件", file=sys.stderr)
        return 1

    processed_path = Path(processed_dataset_path)
    processed_path.mkdir(parents=True, exist_ok=True)

    norm_path = Path(norm_stats_path)
    norm_path.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(checkpoint_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    evaluation_dir = Path(evaluation_output_path)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "raw_dataset_path": str(raw_path.resolve()),
        "processed_dataset_path": str(processed_path.resolve()),
        "norm_stats_path": str(norm_path.resolve()),
        "checkpoint_path": str(checkpoint_dir.resolve()),
        "evaluation_output_path": str(evaluation_dir.resolve()),
    }

    out = ROOT / ".claude" / "artifacts" / "config" / "data_paths.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成配置: {out.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenPI Claude 编排辅助工具")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("show-plan", help="打印 phase/task 编排")

    check = sub.add_parser("check-gate", help="检查 gate 所需产物")
    check.add_argument("gate_id", type=str)
    check.add_argument("--details", action="store_true", help="输出产物的详细检查信息")
    check.add_argument("--json", dest="as_json", action="store_true", help="以 JSON 格式输出校验结果")

    sub.add_parser("dry-run", help="生成 dry-run 任务状态文件")
    sub.add_parser("ensure-artifacts", help="创建 artifacts 目录结构")
    phase_plan = sub.add_parser("phase-plan", help="导出 phase 的依赖分层 dispatch 包")
    phase_plan.add_argument("phase_id", type=str)

    phase_stat = sub.add_parser("phase-status", help="检查 phase 任务完成状态（按产物存在性）")
    phase_stat.add_argument("phase_id", type=str)

    next_task_parser = sub.add_parser("next-tasks", help="给定已完成任务，输出当前可执行任务")
    next_task_parser.add_argument("phase_id", type=str)
    next_task_parser.add_argument(
        "--completed",
        type=str,
        default="",
        help="逗号分隔的已完成 task_id 列表，例如 t1_1,t1_2",
    )

    collect = sub.add_parser("collect-user-config", help="收集 phase_0_5 的路径配置")
    collect.add_argument("--raw-dataset-path", dest="raw_dataset_path", type=str)
    collect.add_argument("--processed-dataset-path", dest="processed_dataset_path", type=str)
    collect.add_argument("--norm-stats-path", dest="norm_stats_path", type=str)
    collect.add_argument("--checkpoint-path", dest="checkpoint_path", type=str)
    collect.add_argument("--evaluation-output-path", dest="evaluation_output_path", type=str)

    args = parser.parse_args()
    cfg = load_config()

    if args.cmd == "show-plan":
        return show_plan(cfg)
    if args.cmd == "check-gate":
        return check_gate(cfg, args.gate_id, details=args.details, as_json=args.as_json)
    if args.cmd == "dry-run":
        return dry_run(cfg)
    if args.cmd == "ensure-artifacts":
        return ensure_artifact_tree()
    if args.cmd == "phase-plan":
        return export_phase_dispatch(cfg, args.phase_id)
    if args.cmd == "phase-status":
        return phase_status(cfg, args.phase_id)
    if args.cmd == "next-tasks":
        completed = [item.strip() for item in args.completed.split(",") if item.strip()]
        return next_tasks(cfg, args.phase_id, completed)
    if args.cmd == "collect-user-config":
        return collect_user_config(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
