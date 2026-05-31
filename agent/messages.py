"""Message formatting and conversation utilities."""
from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.prompts import FACT_GUARDS, SYSTEM_MESSAGE

MAX_HISTORY_MESSAGES = 10
MAX_MESSAGE_CHARS = 4000

UI_NOISE_MARKERS = (
    "lab vitals",
    "gray matter labs",
    "welcome to gray matter",
    "suggested prompt",
    "typing...",
    "loading...",
    "__mock__",
    "tool descriptions",
    "sidebar",
    "ui card",
)


def build_system_prompt() -> str:
    return f"{SYSTEM_MESSAGE}\n\n{FACT_GUARDS}"


def _normalize_role(role: str) -> str | None:
    role = (role or "").strip().lower()
    if role in ("user", "human"):
        return "user"
    if role in ("assistant", "ai", "agent"):
        return "assistant"
    return None


def _is_ui_noise(content: str) -> bool:
    lower = content.strip().lower()
    if not lower:
        return True
    if lower.startswith("{") and len(content) > 500:
        return True
    return any(marker in lower for marker in UI_NOISE_MARKERS)


def _is_skippable_content(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return True
    lower = stripped.lower()
    if lower in ("loading", "loading...", "typing...", "error", "failed"):
        return True
    if lower.startswith("[error") or lower.startswith("error:"):
        return True
    return _is_ui_noise(stripped)


def message_to_dict(message) -> dict | None:
    if isinstance(message, dict):
        role = _normalize_role(message.get("role", ""))
        content = str(message.get("content") or "").strip()
    elif isinstance(message, HumanMessage):
        role, content = "user", str(message.content or "").strip()
    elif isinstance(message, AIMessage):
        role, content = "assistant", str(message.content or "").strip()
    else:
        return None

    if role is None or _is_skippable_content(content):
        return None

    return {"role": role, "content": content[:MAX_MESSAGE_CHARS]}


def dedupe_consecutive_messages(messages: list[dict]) -> list[dict]:
    if not messages:
        return messages
    deduped = [messages[0]]
    for message in messages[1:]:
        prev = deduped[-1]
        if message["role"] == prev["role"] and message["content"] == prev["content"]:
            continue
        deduped.append(message)
    return deduped


def format_messages_for_groq(
    system_prompt: str,
    history: list,
    user_input: str,
) -> list[dict]:
    formatted_history = []
    for message in history:
        normalized = message_to_dict(message)
        if normalized is not None:
            formatted_history.append(normalized)

    formatted_history = dedupe_consecutive_messages(formatted_history)
    formatted_history = formatted_history[-MAX_HISTORY_MESSAGES:]

    current_input = str(user_input or "").strip()[:MAX_MESSAGE_CHARS]
    if formatted_history and formatted_history[-1]["role"] == "user":
        if formatted_history[-1]["content"] == current_input:
            formatted_history = formatted_history[:-1]

    return [
        {"role": "system", "content": system_prompt},
        *formatted_history,
        {"role": "user", "content": current_input},
    ]


def dicts_to_langchain_messages(messages: list[dict]) -> list:
    result = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            result.append(SystemMessage(content=content))
        elif role == "user":
            result.append(HumanMessage(content=content))
        elif role == "assistant":
            result.append(AIMessage(content=content))
    return result


def prepare_messages(messages):
    """Keep only valid user/assistant turns; drop client system/UI noise."""
    prepared = []
    for message in messages:
        if isinstance(message, SystemMessage):
            continue
        normalized = message_to_dict(message)
        if normalized is not None:
            if isinstance(message, HumanMessage):
                prepared.append(HumanMessage(content=normalized["content"]))
            elif isinstance(message, AIMessage):
                prepared.append(AIMessage(content=normalized["content"]))
    return prepared


def is_heisenberg_name_response(message: str) -> bool:
    if not message or not message.strip():
        return False
    low = message.strip().lower()
    if re.fullmatch(r"heisenberg[!.\s]*", low):
        return True
    if re.fullmatch(r"it'?s\s+heisenberg[!.\s]*", low):
        return True
    if re.fullmatch(r"my name is heisenberg[!.\s]*", low):
        return True
    if re.fullmatch(r"say\s+my\s+name\s*:?\s*heisenberg[!.\s]*", low):
        return True
    return False


def extract_conversation(raw_messages: list) -> tuple[list[dict], str | None]:
    conversation = []
    for message in raw_messages:
        normalized = message_to_dict(message)
        if normalized is not None:
            conversation.append(normalized)

    last_user = None
    for message in reversed(conversation):
        if message["role"] == "user":
            last_user = message["content"]
            break

    prior = [
        m
        for m in conversation
        if not (m["role"] == "user" and m["content"] == last_user)
    ] if last_user else conversation

    return prior, last_user
