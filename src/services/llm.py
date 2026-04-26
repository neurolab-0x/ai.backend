import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from src.config.settings import OPENROUTER_CONFIG

logger = logging.getLogger(__name__)


class AsyncOpenRouterClient:
    def __init__(self):
        self.api_key = OPENROUTER_CONFIG["api_key"]
        self.base_url = OPENROUTER_CONFIG["base_url"].rstrip("/")
        self.model = OPENROUTER_CONFIG["model"]
        self.temperature = OPENROUTER_CONFIG["temperature"]
        self.max_tokens = OPENROUTER_CONFIG["max_tokens"]
        self.timeout = OPENROUTER_CONFIG["timeout"]
        self.app_name = OPENROUTER_CONFIG["app_name"]
        self.site_url = OPENROUTER_CONFIG["site_url"]

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": self.app_name,
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        return headers

    async def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self.enabled:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")

        body: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if response_format is not None:
            body["response_format"] = response_format

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=body,
            )
            response.raise_for_status()
            payload = response.json()

        try:
            content = payload["choices"][0]["message"]["content"]
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = [
                    str(item.get("text", "")).strip()
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                return "\n".join(part for part in parts if part).strip()
            return str(content).strip()
        except (KeyError, IndexError, AttributeError) as exc:
            logger.error("Unexpected OpenRouter response format: %s", json.dumps(payload)[:1000])
            raise RuntimeError("Invalid OpenRouter response payload") from exc


_client: Optional[AsyncOpenRouterClient] = None


def get_async_llm_client() -> AsyncOpenRouterClient:
    global _client
    if _client is None:
        _client = AsyncOpenRouterClient()
    return _client
