import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.services.database import db_service
from src.services.llm import get_async_llm_client

logger = logging.getLogger(__name__)

def normalize_history(history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "")).strip().lower()
        if role not in {"user", "assistant", "system"}:
            continue

        content = str(item.get("content", "")).strip()
        if not content:
            continue

        normalized.append({"role": role, "content": content})

    return normalized


def offline_title(history: List[Dict[str, str]], current_title: Optional[str] = None) -> str:
    latest_user = next(
        (entry["content"] for entry in reversed(history) if entry["role"] == "user"),
        "",
    )
    words = [word for word in latest_user.replace("\n", " ").split(" ") if word][:6]
    if words:
        return " ".join(words)
    return current_title or "New Conversation"


def offline_reply(
    message: str,
    history: List[Dict[str, str]],
    include_health_data: bool = True,
) -> str:
    latest_topics = ", ".join(
        entry["content"][:40]
        for entry in history[-3:]
        if entry["role"] == "user"
    )
    if include_health_data:
        return (
            "I am running in offline mode, but I can still help you reason through your neural health data. "
            f"You asked: \"{message.strip()}\". Recent topics in this conversation include: {latest_topics or 'this new request'}."
        )
    return (
        "I am running in offline mode, but I can still help with general discussion. "
        f"You asked: \"{message.strip()}\"."
    )


async def retrieve_chat_context(
    *,
    subject_id: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    include_health_data: bool = True,
    limit: int = 8,
) -> Dict[str, Any]:
    normalized_history = normalize_history(history)
    context: Dict[str, Any] = {
        "history": normalized_history,
        "history_excerpt": normalized_history[-12:],
        "health_history": [],
        "health_context": "",
    }

    if not include_health_data or not subject_id:
        return context

    try:
        user_history = await db_service.get_user_history(subject_id, limit=limit)
        context["health_history"] = user_history
        history_lines = []
        for item in user_history:
            timestamp = item.get("time")
            time_label = (
                timestamp.strftime("%Y-%m-%d %H:%M")
                if hasattr(timestamp, "strftime")
                else str(timestamp)
            )
            if item.get("type") == "session":
                history_lines.append(
                    f"- {time_label}: session {item.get('session_id')} dominant state {item.get('dominant_state')}"
                )
            elif item.get("type") == "training":
                history_lines.append(
                    f"- {time_label}: training run {item.get('run_id')} accuracy {item.get('accuracy')}, loss {item.get('loss')}"
                )
        context["health_context"] = "\n".join(history_lines)
    except Exception as exc:
        logger.warning(f"Failed to load historical context for chat: {exc}")

    return context


async def generate_conversation_title(
    history: List[Dict[str, str]],
    subject_id: Optional[str] = None,
    current_title: Optional[str] = None,
) -> str:
    history = normalize_history(history)
    if not history:
        return current_title or "New Conversation"

    client = get_async_llm_client()
    if not client.enabled:
        return offline_title(history, current_title)

    transcript = "\n".join(
        f"{item['role'].capitalize()}: {item['content']}" for item in history[-8:]
    )
    prompt = (
        "Generate a concise conversation title of at most 6 words for this user conversation. "
        "Return only the title.\n\n"
        f"Subject ID: {subject_id or 'unknown'}\n"
        f"Current Title: {current_title or 'New Conversation'}\n"
        f"{transcript}"
    )

    try:
        title = await client.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=32,
        )
        return title or current_title or "New Conversation"
    except Exception as exc:
        logger.error(f"Failed to generate conversation title: {exc}")
        return offline_title(history, current_title)


async def generate_chat_exchange(
    *,
    message: str,
    subject_id: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    current_title: Optional[str] = None,
    include_health_data: bool = True,
    retrieval_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    context = retrieval_context or await retrieve_chat_context(
        subject_id=subject_id,
        history=history,
        include_health_data=include_health_data,
    )
    normalized_history = context["history"]
    history_context = context["health_context"]
    client = get_async_llm_client()

    if not client.enabled:
        response = offline_reply(
            message=message,
            history=normalized_history,
            include_health_data=include_health_data,
        )
    else:
        system_prompt = (
            "You are the NeuroLab AI assistant. Provide concise, practical, scientifically grounded responses "
            "about EEG, cognition, sleep, stress, and neural health."
        )
        if history_context:
            system_prompt += (
                "\nUse the following prior user context if it is relevant to the latest question:\n"
                f"{history_context}"
            )

        chat_messages = [{"role": "system", "content": system_prompt}]
        chat_messages.extend(normalized_history[-12:])
        chat_messages.append({"role": "user", "content": message.strip()})

        response = await client.create_chat_completion(
            messages=chat_messages,
            temperature=0.5,
            max_tokens=700,
        )

    title_history = [
        *normalized_history,
        {"role": "user", "content": message.strip()},
        {"role": "assistant", "content": response},
    ]
    suggested_title = await generate_conversation_title(
        title_history,
        subject_id=subject_id,
        current_title=current_title,
    )

    return {
        "response": response,
        "suggested_title": suggested_title,
        "should_update_title": bool(suggested_title and suggested_title != (current_title or "")),
        "retrieved_context": {
            "history_items": len(normalized_history),
            "health_history_items": len(context["health_history"]),
            "include_health_data": include_health_data,
        },
        "generated_at": datetime.now().isoformat(),
    }
