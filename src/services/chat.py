import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.services.database import db_service
from src.services.llm import get_async_llm_client
from src.services.user_context import fetch_user_context

logger = logging.getLogger(__name__)

MAX_HISTORY_ITEMS = 20
MAX_HISTORY_CONTENT_CHARS = 4000


def require_llm_client():
    client = get_async_llm_client()
    if not client.enabled:
        raise RuntimeError("LLM chat is unavailable: OPENROUTER_API_KEY is not configured")
    return client

def normalize_history(history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for item in (history or [])[-MAX_HISTORY_ITEMS:]:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "")).strip().lower()
        if role not in {"user", "assistant", "system"}:
            continue

        content = str(item.get("content", "")).strip()[:MAX_HISTORY_CONTENT_CHARS]
        if not content:
            continue

        normalized.append({"role": role, "content": content})

    return normalized

async def retrieve_chat_context(
    *,
    subject_id: Optional[str] = None,
    auth_token: Optional[str] = None,
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
        "user_profile": None,
    }

    if not include_health_data or not subject_id:
        return context

    if auth_token:
        backend_context = await fetch_user_context(
            auth_token=auth_token,
            subject_id=subject_id,
            analysis_limit=limit,
        )
        user_profile = backend_context.get("user")
        analyses = backend_context.get("analyses") or []
        context["user_profile"] = user_profile
        context["health_history"] = analyses

        profile_lines = []
        if user_profile:
            profile_lines.extend(
                [
                    f"- User full name: {user_profile.get('fullName') or 'unknown'}",
                    f"- Username: {user_profile.get('username') or 'unknown'}",
                    f"- Role: {user_profile.get('role') or 'unknown'}",
                ]
            )
            if user_profile.get("email"):
                profile_lines.append(f"- Email: {user_profile['email']}")
            if user_profile.get("phone"):
                profile_lines.append(f"- Phone: {user_profile['phone']}")

        history_lines = []
        for item in analyses:
            history_lines.append(
                f"- {item.get('timestamp')}: analysis status {item.get('status')}, "
                f"notes {item.get('aiNotes') or 'none'}, results {item.get('results') or {}}"
            )

        context["health_context"] = "\n".join([*profile_lines, *history_lines]).strip()
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

    client = require_llm_client()

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
        raise RuntimeError("LLM title generation failed") from exc


async def generate_chat_exchange(
    *,
    message: str,
    subject_id: Optional[str] = None,
    auth_token: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    current_title: Optional[str] = None,
    include_health_data: bool = True,
    retrieval_context: Optional[Dict[str, Any]] = None,
    generate_title: bool = False,
) -> Dict[str, Any]:
    context = retrieval_context or await retrieve_chat_context(
        subject_id=subject_id,
        auth_token=auth_token,
        history=history,
        include_health_data=include_health_data,
    )
    normalized_history = context["history"]
    history_context = context["health_context"]
    client = require_llm_client()
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

    try:
        response = await client.create_chat_completion(
            messages=chat_messages,
            temperature=0.5,
            max_tokens=700,
        )
    except Exception as exc:
        logger.error(f"Failed to generate chat response: {exc}")
        raise RuntimeError("LLM chat generation failed") from exc

    suggested_title = current_title
    should_update_title = False
    if generate_title:
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
        should_update_title = bool(suggested_title and suggested_title != (current_title or ""))

    return {
        "response": response,
        "suggested_title": suggested_title,
        "should_update_title": should_update_title,
        "retrieved_context": {
            "history_items": len(normalized_history),
            "health_history_items": len(context["health_history"]),
            "include_health_data": include_health_data,
        },
        "generated_at": datetime.now().isoformat(),
    }
