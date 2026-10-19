"""Thin wrapper over an OpenAI-compatible chat API (OpenRouter by default).

Three jobs, all of them things a framework would hide from you:
- `chat`: one call with optional tools; retries on rate limits and on OpenRouter's
  "200 with an error body" quirk; counts tokens, calls and cost for the run.
- `structured`: ask for JSON matching a pydantic model, validate it, and on a
  validation error send the errors back and ask again (bounded retries).
- a replay cache: every response is stored on disk under a hash of the request,
  and with LLM_REPLAY=1 an identical request is answered from disk. That makes a
  rehearsed demo reproducible even when the free tier is rate-limited.
"""

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.config import Settings

T = TypeVar("T", bound=BaseModel)

RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4


class LLMError(RuntimeError):
    """A provider failure that survived retries. Carries only the provider's message."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string as the model produced it; parsing is the caller's job


@dataclass
class ChatTurn:
    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str | None
    # The assistant message exactly as it must be appended to the history, so the
    # next call can answer each tool_call_id with a role="tool" message.
    message: dict[str, Any]


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cached_calls: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.calls,
            "cached_calls": self.cached_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 5),
        }


@dataclass
class LLM:
    client: AsyncOpenAI
    model: str
    temperature: float = 0.2
    reasoning: str = "off"
    cache_dir: Path | None = None
    replay: bool = False
    usage: Usage = field(default_factory=Usage)

    @classmethod
    def from_settings(cls, settings: Settings) -> "LLM":
        if not settings.api_key or not settings.model_name:
            raise RuntimeError("API_KEY and MODEL_NAME must be set (see .env.example)")
        client = AsyncOpenAI(base_url=settings.base_url, api_key=settings.api_key, timeout=90)
        return cls(
            client=client,
            model=settings.model_name,
            temperature=settings.temperature,
            reasoning=settings.reasoning,
            cache_dir=settings.cache_dir / "llm",
            replay=settings.llm_replay,
        )

    def for_run(self) -> "LLM":
        """Same client and settings, fresh usage counters: one per user request."""
        return LLM(
            client=self.client,
            model=self.model,
            temperature=self.temperature,
            reasoning=self.reasoning,
            cache_dir=self.cache_dir,
            replay=self.replay,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 2500,
        response_format: dict[str, Any] | None = None,
    ) -> ChatTurn:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            request["tools"] = tools
            request["tool_choice"] = tool_choice or "auto"
        if response_format:
            request["response_format"] = response_format
        # OpenRouter extensions travel in extra_body: reasoning on/off and cost accounting.
        effort = {"enabled": False} if self.reasoning == "off" else {"effort": self.reasoning}
        extra_body = {"reasoning": effort, "usage": {"include": True}}

        raw = self._cache_get(request) if self.replay else None
        if raw is not None:
            self.usage.cached_calls += 1
        else:
            raw = await self._call_with_retries(request, extra_body)
            self._cache_put(request, raw)
            self._count(raw)
        return _parse_turn(raw)

    async def structured(
        self,
        messages: list[dict[str, Any]],
        schema: type[T],
        max_retries: int = 2,
        max_tokens: int = 2500,
    ) -> T:
        """Return a validated `schema` instance; feed validation errors back on failure."""
        json_schema = _require_all(schema.model_json_schema())
        history = [
            *messages,
            {
                "role": "system",
                "content": "Reply with a single JSON object that matches this JSON Schema. "
                "No prose, no markdown fences.\n" + json.dumps(json_schema, ensure_ascii=False),
            },
        ]
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": json_schema, "strict": False},
        }
        last_error = ""
        for _ in range(max_retries + 1):
            turn = await self.chat(
                history, temperature=0, max_tokens=max_tokens, response_format=response_format
            )
            text = turn.content or ""
            try:
                return schema.model_validate(json.loads(_extract_json(text)))
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = str(e)[:1500]
                history = [
                    *history,
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": "That JSON did not validate against the schema:\n"
                        f"{last_error}\nReturn the corrected JSON object only.",
                    },
                ]
        raise LLMError(f"model did not return valid {schema.__name__} JSON: {last_error}")

    async def _call_with_retries(
        self, request: dict[str, Any], extra_body: dict[str, Any]
    ) -> dict[str, Any]:
        delay = 2.0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = await self.client.chat.completions.create(**request, extra_body=extra_body)
                raw = resp.model_dump()
                # OpenRouter may answer 200 with {"error": {...}} and no choices when the
                # upstream provider is busy; treat it like a 502 and retry.
                if not raw.get("choices"):
                    err = (raw.get("error") or {}) if isinstance(raw.get("error"), dict) else {}
                    raise LLMError(err.get("message") or "provider returned no choices", 502)
                return raw
            except APIStatusError as e:
                status = e.status_code
                body = e.body if isinstance(e.body, dict) else {}
                # Pass on only the provider's message: the raw body carries the account id.
                message = (body.get("error") or {}).get("message") or str(e.message)
                if status not in RETRY_STATUSES or attempt == MAX_ATTEMPTS:
                    raise LLMError(message, status) from e
            except (APIConnectionError, APITimeoutError) as e:
                if attempt == MAX_ATTEMPTS:
                    raise LLMError(f"connection to the LLM provider failed: {e}", 504) from e
            except LLMError:
                if attempt == MAX_ATTEMPTS:
                    raise
            await asyncio.sleep(delay)
            delay *= 2
        raise LLMError("unreachable")

    def _count(self, raw: dict[str, Any]) -> None:
        usage = raw.get("usage") or {}
        self.usage.calls += 1
        self.usage.prompt_tokens += usage.get("prompt_tokens") or 0
        self.usage.completion_tokens += usage.get("completion_tokens") or 0
        self.usage.cost_usd += float(usage.get("cost") or 0.0)

    def _cache_key(self, request: dict[str, Any]) -> str:
        blob = json.dumps(request, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_get(self, request: dict[str, Any]) -> dict[str, Any] | None:
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"{self._cache_key(request)}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def _cache_put(self, request: dict[str, Any], raw: dict[str, Any]) -> None:
        if not self.cache_dir:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{self._cache_key(request)}.json"
        path.write_text(json.dumps(raw, ensure_ascii=False))


def _parse_turn(raw: dict[str, Any]) -> ChatTurn:
    choice = raw["choices"][0]
    msg = choice.get("message") or {}
    calls = [
        ToolCall(
            id=tc["id"],
            name=tc["function"]["name"],
            arguments=tc["function"].get("arguments") or "{}",
        )
        for tc in (msg.get("tool_calls") or [])
    ]
    message: dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
    if calls:
        message["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in calls
        ]
    return ChatTurn(
        content=msg.get("content"),
        tool_calls=calls,
        finish_reason=choice.get("finish_reason"),
        message=message,
    )


def _require_all(json_schema: dict[str, Any]) -> dict[str, Any]:
    """List every property as required; nullable ones may still be null.

    Providers compile the schema into a grammar, and optional properties make that
    expensive: a 15-field all-optional schema stalled for minutes on Claude via
    OpenRouter, while the same schema with every field required answered in seconds.
    Validation is unchanged: the pydantic model keeps its defaults.
    """
    for node in [json_schema, *json_schema.get("$defs", {}).values()]:
        if "properties" in node:
            node["required"] = list(node["properties"])
    return json_schema


def _extract_json(text: str) -> str:
    """Models sometimes wrap JSON in ``` fences or add a sentence; take the object."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text
