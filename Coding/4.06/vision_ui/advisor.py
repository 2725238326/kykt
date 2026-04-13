from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from job_store import ROOT, get_job_dir, get_log_snippets, iter_input_items, load_job, load_result_summary


SETTINGS_DIR = ROOT / "settings"
SETTINGS_PATH = SETTINGS_DIR / "advisor.json"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_SYSTEM_PROMPT = """你是 KYKT Vision 项目的实验评估助手。

你的任务是根据任务模型、输入数量、关键参数、结果摘要、scene_meta、最近日志等信息，对三维重建任务做简洁而具体的判断。

请始终输出 JSON 对象，字段固定为：
- overall_score: 1 到 10 的整数分
- readiness: unusable / exploratory / usable / strong
- summary: 一段 2 到 4 句的总体判断
- issues: 字符串数组，列出最关键的问题
- next_actions: 字符串数组，列出最值得立刻执行的下一步
- teacher_talk: 一段适合向老师口头汇报的简短话术

要求：
- 不要空泛鼓励，要结合任务上下文。
- 如果结果只是链路验证，也要明确说清。
- 如果日志显示未配置 API 或任务失败，就直接点出阻塞项。
""".strip()

DEFAULT_CONFIG = {
    "enabled": False,
    "base_url": "",
    "api_key": "",
    "model": DEFAULT_MODEL,
    "temperature": 0.2,
    "max_tokens": 1200,
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
}


def load_advisor_config() -> dict[str, Any]:
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    if not SETTINGS_PATH.exists():
        return dict(DEFAULT_CONFIG)

    payload = json.loads(SETTINGS_PATH.read_text(encoding="utf-8-sig"))
    merged = dict(DEFAULT_CONFIG)
    merged.update(payload)
    return merged


def advisor_config_public() -> dict[str, Any]:
    config = load_advisor_config()
    status = advisor_status()
    return {
        **status,
        "has_api_key": bool(str(config.get("api_key") or "").strip()),
        "temperature": float(config.get("temperature") or 0.2),
        "max_tokens": int(config.get("max_tokens") or 1200),
        "system_prompt": str(config.get("system_prompt") or DEFAULT_SYSTEM_PROMPT),
    }


def save_advisor_config(payload: dict[str, Any]) -> dict[str, Any]:
    current = load_advisor_config()
    merged = dict(DEFAULT_CONFIG)
    merged.update(current)

    if "enabled" in payload:
        merged["enabled"] = bool(payload.get("enabled"))
    if "base_url" in payload:
        merged["base_url"] = str(payload.get("base_url") or "").strip()
    if "model" in payload:
        merged["model"] = str(payload.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    if "temperature" in payload:
        merged["temperature"] = float(payload.get("temperature") or 0.2)
    if "max_tokens" in payload:
        merged["max_tokens"] = int(payload.get("max_tokens") or 1200)
    if "system_prompt" in payload:
        merged["system_prompt"] = str(payload.get("system_prompt") or DEFAULT_SYSTEM_PROMPT).strip() or DEFAULT_SYSTEM_PROMPT

    if "api_key" in payload:
        api_key = str(payload.get("api_key") or "").strip()
        if api_key:
            merged["api_key"] = api_key
        elif payload.get("clear_api_key"):
            merged["api_key"] = ""

    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    return advisor_config_public()


def advisor_status() -> dict[str, Any]:
    config = load_advisor_config()
    base_url = str(config.get("base_url") or "").strip()
    api_key = str(config.get("api_key") or "").strip()
    model = str(config.get("model") or "").strip()
    enabled = bool(config.get("enabled"))
    configured = bool(base_url and api_key and model)
    return {
        "enabled": enabled,
        "configured": configured,
        "base_url": base_url,
        "model": model or DEFAULT_MODEL,
        "has_api_key": bool(api_key),
        "message": _advisor_status_message(enabled, configured),
    }


def load_advisor_report(job_id: str) -> dict[str, Any] | None:
    path = get_job_dir(job_id) / "advisor_report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_advisor_report(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = get_job_dir(job_id) / "advisor_report.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def evaluate_job_with_advisor(job_id: str) -> dict[str, Any]:
    status = advisor_status()
    if not status["enabled"]:
        raise RuntimeError("AI 评估尚未启用。请先编辑 settings/advisor.json，把 enabled 改成 true，并填入 base_url、api_key、model。")
    if not status["configured"]:
        raise RuntimeError("AI 评估配置不完整。请检查 settings/advisor.json 里的 base_url、api_key、model。")

    config = load_advisor_config()
    context = build_advisor_context(job_id)
    raw_text = _call_openai_compatible_api(config, context)
    parsed = _parse_model_json(raw_text)

    report = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "job_id": job_id,
        "overall_score": int(parsed.get("overall_score") or 0),
        "readiness": str(parsed.get("readiness") or "exploratory"),
        "summary": str(parsed.get("summary") or "").strip(),
        "issues": _normalize_text_list(parsed.get("issues")),
        "next_actions": _normalize_text_list(parsed.get("next_actions")),
        "teacher_talk": str(parsed.get("teacher_talk") or "").strip(),
        "advisor_model": str(config.get("model") or DEFAULT_MODEL),
    }
    return save_advisor_report(job_id, report)


def build_advisor_context(job_id: str) -> dict[str, Any]:
    job = load_job(job_id)
    summary = load_result_summary(job_id)
    scene_meta = summary.get("scene_meta") if summary else None
    logs = get_log_snippets(job_id, limit=80)

    return {
        "project": "KYKT Vision",
        "job": {
            "job_id": job.job_id,
            "model": job.model,
            "source_type": job.source_type,
            "created_at": job.created_at,
            "status": job.status,
            "phase": job.phase,
            "notes": job.notes,
            "params": job.params,
            "progress_message": job.progress_message,
            "error_message": job.error_message,
            "input_count": len(job.input_files),
            "input_names": [item["original_name"] for item in iter_input_items(job)],
            "output_files": job.output_files,
        },
        "result_summary": summary,
        "scene_meta": scene_meta,
        "logs": [
            {
                "name": item["name"],
                "tail": item["tail"],
            }
            for item in logs
        ],
        "expected_output_format": {
            "overall_score": "1-10 integer",
            "readiness": "unusable | exploratory | usable | strong",
            "summary": "brief Chinese summary",
            "issues": "string array",
            "next_actions": "string array",
            "teacher_talk": "brief Chinese oral update",
        },
    }


def _advisor_status_message(enabled: bool, configured: bool) -> str:
    if not enabled:
        return "AI 评估已接入，但当前处于关闭状态。"
    if not configured:
        return "AI 评估已启用，但配置不完整。"
    return "AI 评估已就绪。"


def _call_openai_compatible_api(config: dict[str, Any], context: dict[str, Any]) -> str:
    base_url = str(config.get("base_url") or "").rstrip("/")
    if base_url.endswith("/chat/completions"):
        url = base_url
    elif base_url.endswith("/v1"):
        url = f"{base_url}/chat/completions"
    else:
        url = f"{base_url}/chat/completions"

    payload = {
        "model": str(config.get("model") or DEFAULT_MODEL),
        "temperature": float(config.get("temperature") or 0.2),
        "max_tokens": int(config.get("max_tokens") or 1200),
        "messages": [
            {"role": "system", "content": str(config.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)},
            {
                "role": "user",
                "content": (
                    "请严格返回 JSON 对象，不要加 Markdown 代码块。\n\n"
                    "任务上下文如下：\n"
                    f"{json.dumps(context, ensure_ascii=False, indent=2)}"
                ),
            },
        ],
    }

    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {str(config.get('api_key') or '').strip()}",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=90) as response:
            body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(f"AI 评估请求失败：HTTP {exc.code}。{detail[:400]}") from exc
    except URLError as exc:
        raise RuntimeError(f"AI 评估连接失败：{exc.reason}") from exc

    response_payload = json.loads(body)
    choice = (response_payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                pieces.append(str(item.get("text") or ""))
        return "\n".join(piece for piece in pieces if piece).strip()
    return str(content or "").strip()


def _parse_model_json(raw_text: str) -> dict[str, Any]:
    if not raw_text:
        raise RuntimeError("AI 评估接口返回了空内容。")

    cleaned = raw_text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise RuntimeError(f"AI 评估返回的不是合法 JSON：{cleaned[:400]}")


def _normalize_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        text = str(item or "").strip()
        if text:
            items.append(text)
    return items
