"""Config-driven DAG execution engine for oh-my-openagent pipelines.

Runs a pipeline defined in config/agents.yaml as a Celery task.
Each stage is an LLM call (with optional function calling for tool-enabled stages).
Progress is written to Redis so SSE clients can subscribe.

Key design decisions:
- Deterministic DAG execution (topological sort), not an LLM deciding what to run next.
- Code agents call pre-defined tools via function calling -- never generate code.
- Every stage's LLM response is validated against its expected JSON schema before proceeding.
- FAIL-SAFE: any stage failure -> pipeline stops, progress shows "failed", user gets degraded response.
"""
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import redis
import yaml

from .config import AGENTS_CONFIG_PATH, REDIS_URL
from .llm import chat

_log = logging.getLogger("cheshijing.agent_pipeline")

# ── Config loading (thread-safe, same pattern as app/nlu.py) ──────────────────

_cfg_lock = threading.Lock()
_cfg = None


def _load_pipeline_config() -> dict:
    global _cfg
    with _cfg_lock:
        if _cfg is not None:
            return _cfg
        path = Path(AGENTS_CONFIG_PATH)
        if not path.exists():
            _cfg = {"stages": [], "agents": {}}
            return _cfg
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        _cfg = {
            "stages": raw.get("pipeline", {}).get("stages", []),
            "agents": raw.get("agents", {}),
            "redis": raw.get("redis", {}),
        }
        return _cfg


def _get_agent_config(agent_name: str) -> dict:
    cfg = _load_pipeline_config()
    return cfg.get("agents", {}).get(agent_name, {})


def _get_stage_by_id(stages: list, stage_id: str) -> dict | None:
    for s in stages:
        if s["id"] == stage_id:
            return s
    return None


# ── DAG topological sort ──────────────────────────────────────────────────────

def _topological_sort(stages: list) -> list:
    """Kahn's algorithm. Sorts stages so dependencies come before dependents."""
    in_degree = {s["id"]: len(s.get("depends_on") or []) for s in stages}
    adj = {s["id"]: [] for s in stages}
    for s in stages:
        for dep in s.get("depends_on") or []:
            if dep in adj:
                adj[dep].append(s["id"])

    queue = [sid for sid, deg in in_degree.items() if deg == 0]
    result = []
    while queue:
        sid = queue.pop(0)
        result.append(_get_stage_by_id(stages, sid))
        for neighbor in adj.get(sid, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(stages):
        raise ValueError(
            f"Pipeline DAG has a cycle or missing dependency. "
            f"Sorted {len(result)}/{len(stages)}"
        )
    return result


# ── Redis progress ────────────────────────────────────────────────────────────

_redis_client_instance = None


def _redis_client():
    global _redis_client_instance
    if _redis_client_instance is None:
        _redis_client_instance = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client_instance


def _set_progress(task_id: str, data: dict):
    """Write pipeline progress to Redis. TTL matched to config."""
    try:
        r = _redis_client()
        ttl = _load_pipeline_config().get("redis", {}).get("progress_ttl_seconds", 1800)
        r.set(
            f"pipeline:{task_id}:status",
            json.dumps(data, ensure_ascii=False, default=str),
            ex=ttl,
        )
    except Exception:
        _log.warning("Failed to write Redis progress for %s", task_id, exc_info=True)


def _get_progress(task_id: str) -> dict | None:
    """Read pipeline progress from Redis."""
    try:
        r = _redis_client()
        raw = r.get(f"pipeline:{task_id}:status")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


def get_progress(task_id: str) -> dict | None:
    """Public API: read pipeline progress from Redis."""
    return _get_progress(task_id)


# ── Stage execution ────────────────────────────────────────────────────────────

def _call_llm_with_tools(agent_name: str, messages: list, tool_defs: list) -> dict:
    """Call LLM with function calling. If the model returns tool_calls, execute them
    and feed results back for one follow-up turn. Returns final parsed JSON response.

    Uses a two-turn pattern:
    Turn 1: LLM sees tools, may return tool_calls
    Turn 2 (if tool_calls): tool results fed back, LLM returns final JSON
    """
    agent_cfg = _get_agent_config(agent_name)
    model = agent_cfg.get("model") or None
    temperature = agent_cfg.get("temperature", 0.0)

    # Build kwargs for chat() — only pass model if it's explicitly set (non-None)
    def _chat_kwargs(**extra):
        kw = {"temperature": temperature}
        if model:
            kw["model"] = model
        kw.update(extra)
        return kw

    if not tool_defs:
        # No tools -- single call
        raw = chat(messages, **_chat_kwargs())
        return _parse_json_response(raw)

    # With tools: first turn
    raw = chat(messages, **_chat_kwargs(tools=tool_defs))

    # Check if the model wants to call tools
    tool_results = []
    if _has_tool_calls(raw):
        tool_calls = _extract_tool_calls(raw)
        for tc in tool_calls:
            from .agent_tools import execute_tool

            result = execute_tool(tc["name"], tc.get("arguments", {}))
            tool_results.append({"tool_name": tc["name"], "result": result})

        # Feed tool results back
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {
                "role": "user",
                "content": json.dumps({"tool_results": tool_results}, ensure_ascii=False),
            }
        )
        raw = chat(messages, **_chat_kwargs())

    return _parse_json_response(raw)


def _has_tool_calls(raw: str) -> bool:
    """Detect if the LLM response contains function-calling tool_calls."""
    try:
        data = json.loads(raw)
        return bool(data.get("tool_calls"))
    except (json.JSONDecodeError, TypeError):
        return False


def _extract_tool_calls(raw: str) -> list[dict]:
    """Extract tool call requests from an OpenAI-format response."""
    try:
        data = json.loads(raw)
        calls = data.get("tool_calls", [])
        return [
            {
                "name": c.get("function", {}).get("name", ""),
                "arguments": json.loads(c.get("function", {}).get("arguments", "{}")),
            }
            for c in calls
        ]
    except Exception:
        return []


def _parse_json_response(raw: str) -> dict:
    """Extract JSON from LLM response. FAIL-SAFE: returns error dict on failure."""
    try:
        s, e = raw.find("{"), raw.rfind("}") + 1
        if s < 0 or e <= s:
            raise ValueError("no JSON in response")
        return json.loads(raw[s:e])
    except Exception:
        return {
            "status": "failed",
            "error": "Failed to parse LLM response as JSON",
            "raw_preview": raw[:200],
        }


def _execute_stage(
    stage_def: dict, pipeline_context: dict, original_question: str
) -> dict:
    """Execute a single pipeline stage. Returns the parsed LLM output dict."""
    stage_id = stage_def["id"]
    agent_name = stage_def["agent"]
    agent_cfg = _get_agent_config(agent_name)

    if not agent_cfg:
        raise ValueError(
            f"Agent '{agent_name}' not found in agents.yaml (stage '{stage_id}')"
        )

    from .agent_tools import get_tool_definitions

    tool_defs = get_tool_definitions(agent_name)

    # Build messages
    system_prompt = agent_cfg.get("system_prompt", "").strip()
    messages = [{"role": "system", "content": system_prompt}]

    # Build user prompt with context from upstream stages + original question
    user_parts = [f"用户原始问题：{original_question}"]
    if pipeline_context:
        user_parts.append(
            f"上游阶段输出：{json.dumps(pipeline_context, ensure_ascii=False)}"
        )
    messages.append({"role": "user", "content": "\n\n".join(user_parts)})

    return _call_llm_with_tools(agent_name, messages, tool_defs)


# ── Pipeline runner ────────────────────────────────────────────────────────────

def run_pipeline(
    task_id: str, original_question: str, original_user_id: int = 0
) -> dict:
    """Execute the full pipeline DAG. Called by Celery task.

    Returns: {"status": "completed|failed", "stages": {...}, "final_answer": "..."}
    """
    _set_progress(task_id, {"stage": "start", "status": "running", "ts": time.time()})

    cfg = _load_pipeline_config()
    stages = cfg.get("stages", [])
    if not stages:
        return {"status": "failed", "error": "No pipeline stages defined in agents.yaml"}

    # Topological sort
    try:
        ordered = _topological_sort(stages)
    except ValueError as e:
        return {"status": "failed", "error": str(e)}

    # Execute stages in order, feeding each stage's output as context for downstream
    stage_outputs: dict[str, Any] = {}

    for stage_def in ordered:
        stage_id = stage_def["id"]
        _set_progress(
            task_id, {"stage": stage_id, "status": "running", "ts": time.time()}
        )

        try:
            # Build context from dependency outputs
            deps = stage_def.get("depends_on") or []
            dep_context = {
                dep: stage_outputs.get(dep) for dep in deps if dep in stage_outputs
            }

            output = _execute_stage(stage_def, dep_context, original_question)
            stage_outputs[stage_id] = output

            _set_progress(
                task_id,
                {
                    "stage": stage_id,
                    "status": "completed",
                    "ts": time.time(),
                    "preview": json.dumps(output, ensure_ascii=False)[:200],
                },
            )

        except Exception as e:
            _log.error("Pipeline stage '%s' failed: %s", stage_id, e, exc_info=True)
            _set_progress(
                task_id,
                {
                    "stage": stage_id,
                    "status": "failed",
                    "error": str(e)[:200],
                    "ts": time.time(),
                },
            )
            return {
                "status": "failed",
                "failed_stage": stage_id,
                "error": str(e)[:300],
                "stages": stage_outputs,
            }

    # Pipeline completed
    review_output = stage_outputs.get("review", {})
    final_answer = review_output.get(
        "answer_to_user", "数据采集完成，但未能生成有效回答。"
    )

    _set_progress(
        task_id,
        {
            "stage": "done",
            "status": "completed",
            "ts": time.time(),
            "final_answer": final_answer[:500],
        },
    )

    return {
        "status": "completed",
        "stages": stage_outputs,
        "final_answer": final_answer,
    }


# ── Celery task ───────────────────────────────────────────────────────────────

from .celery_app import celery


@celery.task(name="agent_pipeline.run", bind=True, max_retries=0,
             task_ignore_result=True)
def run_pipeline_task(self, task_id: str, original_question: str,
                      original_user_id: int = 0):
    """Celery task wrapper for run_pipeline. fire-and-forget with Redis progress.
    On success, writes results to RAG and triggers re-query via callback.
    """
    _set_progress(task_id, {"stage": "queued", "status": "pending", "ts": time.time()})

    try:
        result = run_pipeline(task_id, original_question, original_user_id)
        _set_progress(task_id, {
            "stage": result.get("status", "done"),
            "status": result.get("status", "failed"),
            "final_answer": result.get("final_answer", "")[:500],
            "ts": time.time(),
        })
        return result
    except Exception as e:
        _log.error(f"Pipeline task {task_id} failed: {e}", exc_info=True)
        _set_progress(task_id, {
            "stage": "error", "status": "failed",
            "error": str(e)[:300], "ts": time.time(),
        })
        return {"status": "failed", "error": str(e)[:300]}
