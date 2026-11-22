"""OpenAI-compatible LLM adapter used by the bounded task loop."""

from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv


@dataclass(frozen=True)
class ToolCall:
    """A tool call requested by the model."""

    name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclass(frozen=True)
class LLMResponse:
    """The small response shape the orchestration loop needs."""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    cost_inr: float = 0.0


class LLM(Protocol):
    """Protocol allowing the real client to be replaced in tests."""

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], timeout: float
    ) -> LLMResponse: ...


def parse_tool_arguments(raw: str, tool_name: str) -> dict[str, Any]:
    """Parse strict JSON and Groq-compatible Python-literal argument objects."""
    try:
        arguments = json.loads(raw or "{}")
    except json.JSONDecodeError:
        try:
            arguments = ast.literal_eval(raw or "{}")
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"LLM returned invalid tool-call JSON for {tool_name}: {exc}"
            ) from exc
    if not isinstance(arguments, dict):
        raise ValueError(f"LLM returned non-object tool-call arguments for {tool_name}")
    return arguments


def is_malformed_tool_call_error(exc: Exception) -> bool:
    text = str(exc).casefold()
    return "tool_use_failed" in text or "failed to parse tool call arguments" in text


class OpenAICompatibleLLM:
    """Call any OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        load_dotenv(root / "src" / ".env", override=False)
        load_dotenv(root / ".env", override=False)
        self.base_url = self._value(
            base_url,
            "LLM_BASE_URL",
            "OPENAI_BASE_URL",
            "OPENAI_COMPATIBLE_BASE_URL",
        )
        self.model = self._value(model, "LLM_MODEL", "OPENAI_MODEL")
        self.api_key = self._value(
            api_key,
            "LLM_API_KEY",
            "OPENAI_API_KEY",
            "OPENAI_COMPATIBLE_API_KEY",
        )

        if self.base_url and self.base_url.startswith("gsk_"):
            if not self.api_key:
                self.api_key = self.base_url
            self.base_url = "https://api.groq.com/openai/v1"
        elif self.api_key and not self.base_url and self.api_key.startswith("gsk_"):
            self.base_url = "https://api.groq.com/openai/v1"
        if self.base_url and not self.base_url.lower().startswith(("http://", "https://")):
            raise ValueError(
                "Invalid LLM_BASE_URL: expected an http:// or https:// OpenAI-compatible URL."
            )

        missing = [name for name, value in (
            ("LLM_MODEL", self.model),
        ) if not value]
        if not self.base_url:
            missing.append("LLM_BASE_URL")
        if missing:
            raise ValueError(f"Missing LLM configuration: {', '.join(missing)}")
        self.api_key = self.api_key or "not-needed"

    @staticmethod
    def _value(explicit: str | None, *names: str) -> str | None:
        candidates = (explicit, *(os.getenv(name) for name in names))
        for value in candidates:
            if value and value.strip():
                return value.strip().strip("\"'")
        return None

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], timeout: float
    ) -> LLMResponse:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout)
        active_messages = messages
        response = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=active_messages,
                    tools=tools,
                    tool_choice="auto",
                )
                break
            except Exception as exc:
                if is_malformed_tool_call_error(exc) and attempt < 2:
                    active_messages = [
                        *messages,
                        {
                            "role": "system",
                            "content": (
                                "Your previous tool-call arguments were invalid JSON. "
                                "Retry with only strict JSON object syntax: double-quoted "
                                "keys and string values, no trailing commas or Python literals."
                            ),
                        },
                    ]
                    continue
                raise ValueError(f"LLM request failed: {exc}") from exc
        if response is None:
            raise ValueError("LLM request failed: no response returned")
        message = response.choices[0].message
        calls = []
        for call in message.tool_calls or []:
            arguments = parse_tool_arguments(call.function.arguments, call.function.name)
            calls.append(
                ToolCall(
                    name=call.function.name,
                    arguments=arguments,
                    call_id=call.id,
                )
            )
        usage = response.usage
        return LLMResponse(
            content=message.content or "",
            tool_calls=tuple(calls),
            cost_inr=float(getattr(usage, "cost_inr", 0.0) or 0.0),
        )
