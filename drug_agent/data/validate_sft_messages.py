from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))


def _is_tool_response_user_text(text: str) -> bool:
    s = text.strip()
    return s.startswith("<tool_response>") and s.endswith("</tool_response>")


def _preview_messages(messages: Any, limit: int = 6) -> Any:
    if not isinstance(messages, list):
        return messages
    return messages[:limit]


def _roles(messages: Any) -> list[str]:
    if not isinstance(messages, list):
        return []
    out: list[str] = []
    for item in messages:
        if isinstance(item, dict) and isinstance(item.get("role"), str):
            out.append(item["role"])
    return out


def validate_structure(messages: Any) -> list[str]:
    reasons: list[str] = []

    if not isinstance(messages, list) or not messages:
        return ["messages_empty_or_not_list"]

    roles = []
    for msg in messages:
        if not isinstance(msg, dict):
            reasons.append("message_not_object")
            continue
        role = msg.get("role")
        content = msg.get("content")
        roles.append(role)

        if role not in {"system", "user", "assistant"}:
            reasons.append("unsupported_role")
        if not isinstance(content, str):
            reasons.append("content_not_string")
        elif not content.strip():
            reasons.append("content_empty")

    if roles and roles[0] != "system" and roles[0] != "user":
        reasons.append("bad_first_role")

    first_non_system = next((r for r in roles if r != "system"), None)
    if first_non_system is not None and first_non_system != "user":
        reasons.append("first_non_system_not_user")

    has_user = any(
        isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str) and m.get("content").strip()
        for m in messages
    )
    has_assistant = any(
        isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str) and m.get("content").strip()
        for m in messages
    )

    if not has_user:
        reasons.append("no_nonempty_user_turn")
    if not has_assistant:
        reasons.append("no_nonempty_assistant_turn")

    users = [
        m
        for m in messages
        if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str) and m.get("content").strip()
    ]
    if users and not any(not _is_tool_response_user_text(m["content"]) for m in users):
        reasons.append("no_user_query_only_tool_response_users")

    return sorted(set(reasons))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SFT messages for slime/Qwen chat template compatibility")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional HF tokenizer path for apply_chat_template validation")
    parser.add_argument("--preview", type=int, default=20)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            records.append((i, obj))

    bad = []
    reason_counter = Counter()

    for i, obj in records:
        messages = obj.get("messages")
        reasons = validate_structure(messages)
        if reasons:
            for r in reasons:
                reason_counter[r] += 1
            bad.append(
                {
                    "line": i,
                    "index": i,
                    "task_id": (obj.get("metadata") or {}).get("task_id") if isinstance(obj.get("metadata"), dict) else None,
                    "roles": _roles(messages),
                    "reasons": reasons,
                    "preview": _preview_messages(messages),
                }
            )

    apply_template_failed = []
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
            for i, obj in records:
                messages = obj.get("messages")
                try:
                    tok.apply_chat_template(messages, tokenize=False, return_dict=False)
                except Exception as exc:
                    apply_template_failed.append(
                        {
                            "line": i,
                            "index": i,
                            "task_id": (obj.get("metadata") or {}).get("task_id") if isinstance(obj.get("metadata"), dict) else None,
                            "roles": _roles(messages),
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "preview": _preview_messages(messages),
                        }
                    )
        except Exception as exc:
            # Keep tokenizer errors in apply_template_failed so final output remains one structured JSON object.
            apply_template_failed.append(
                {
                    "line": -1,
                    "index": -1,
                    "task_id": None,
                    "roles": [],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "preview": None,
                }
            )

    summary = {
        "ok": len(bad) == 0 and len(apply_template_failed) == 0,
        "input": str(input_path),
        "total": len(records),
        "bad": len(bad),
        "bad_reason_counts": dict(reason_counter),
        "apply_chat_template_failed": len(apply_template_failed),
    }

    bad_preview: list[dict[str, Any]] = []
    for item in bad[: args.preview]:
        bad_preview.append(
            {
                "line": item.get("line"),
                "index": item.get("index"),
                "task_id": item.get("task_id"),
                "roles": item.get("roles"),
                "reason": item.get("reasons"),
                "preview": item.get("preview"),
            }
        )

    apply_failed_preview: list[dict[str, Any]] = []
    for item in apply_template_failed[: args.preview]:
        apply_failed_preview.append(
            {
                "line": item.get("line"),
                "index": item.get("index"),
                "task_id": item.get("task_id"),
                "roles": item.get("roles"),
                "reason": f"{item.get('error_type')}: {item.get('error_message')}",
                "preview": item.get("preview"),
            }
        )

    if bad_preview:
        summary["bad_samples_preview"] = bad_preview
    if apply_failed_preview:
        summary["apply_chat_template_failed_preview"] = apply_failed_preview

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
