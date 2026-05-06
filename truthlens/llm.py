"""Small OpenAI-compatible chat client and JSON parsing helpers."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Protocol


class ChatClient(Protocol):
    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        response_format_json: bool = False,
    ) -> str:
        """Return a chat-completion response as text."""


class OpenAICompatibleChatClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 600,
        retries: int = 6,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        response_format_json: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}

        response = self._post_json(self.base_url + "/chat/completions", payload)
        return response["choices"][0]["message"]["content"]

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        last_detail = ""

        for attempt in range(self.retries):
            request = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "truthlens-memory-audit/0.1",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                last_detail = exc.read().decode("utf-8", errors="replace")
                retry_after = exc.headers.get("retry-after")
                if retry_after and retry_after.isdigit():
                    sleep_seconds = int(retry_after)
                elif exc.code in {408, 409, 429, 500, 502, 503, 504}:
                    sleep_seconds = min(120, 5 * (attempt + 1))
                else:
                    raise RuntimeError(f"API error {exc.code}: {last_detail}") from exc
                time.sleep(sleep_seconds)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(min(120, 5 * (attempt + 1)))

        raise RuntimeError(f"API request failed after retries. Last detail: {last_detail}") from last_error


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {"parse_error": raw[:1200]}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"parse_error": raw[:1200]}
    return parsed if isinstance(parsed, dict) else {"parse_error": raw[:1200]}
