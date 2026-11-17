"""Async client for the Bodhan hosted model APIs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

BASE_URL = "https://api.bodhan.ai/v1"
ModelName = Literal["transcribe", "translate", "speak", "ocr"]


class BodhanError(BaseModel):
    """Actionable error returned by a Bodhan operation."""

    code: str
    message: str
    hint: str


class BodhanResult(BaseModel):
    """Common typed response envelope for model operations."""

    model_config = ConfigDict(extra="allow")

    ok: bool
    model: ModelName
    data: dict[str, Any] = Field(default_factory=dict)
    cost_inr: float = 0.0
    cached: bool = False
    error: BodhanError | None = None


class BodhanClient:
    """Call Bodhan APIs with retries, caching, concurrency, and cost tracking."""

    _keys = {
        "transcribe": "BODHAN_TRANSCRIBE_KEY",
        "translate": "BODHAN_TRANSLATE_KEY",
        "speak": "BODHAN_SPEAK_KEY",
        "ocr": "BODHAN_OCR_KEY",
    }
    _paths = {
        "transcribe": "/audio/transcriptions",
        "translate": "/chat/completions",
        "speak": "/audio/speech",
        "ocr": "/ocr",
    }

    def __init__(
        self,
        *,
        base_url: str = BASE_URL,
        cache_dir: str | Path = ".cache/bodhan",
        timeout_seconds: float = 60.0,
        max_concurrency: int = 4,
        max_retries: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.timeout = httpx.Timeout(timeout_seconds)
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._transport = transport

    async def close(self) -> None:
        """Close the underlying HTTP client if one has been created."""
        if hasattr(self, "_http"):
            await self._http.aclose()

    async def __aenter__(self) -> BodhanClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def transcribe(self, audio_path: str | Path, **options: Any) -> BodhanResult:
        """Transcribe an audio file through Indic-Transcribe."""
        return await self._file_request("transcribe", audio_path, options)

    async def translate(self, text: str, **options: Any) -> BodhanResult:
        """Translate text through Indic-Translate."""
        payload = {"text": text, **options}
        return await self._json_request("translate", payload)

    async def speak(self, text: str, **options: Any) -> BodhanResult:
        """Synthesize speech through Indic-Speak."""
        payload = {"text": text, **options}
        return await self._json_request("speak", payload)

    async def ocr(self, image_path: str | Path, **options: Any) -> BodhanResult:
        """Extract text from an image through Indic-OCR."""
        return await self._file_request("ocr", image_path, options)

    async def _file_request(
        self, model: ModelName, path: str | Path, options: dict[str, Any]
    ) -> BodhanResult:
        file_path = Path(path)
        if not file_path.is_file():
            return self._failure(model, "file_not_found", f"Input file does not exist: {file_path}",
                                 "Provide an existing file path.")
        if file_path.stat().st_size == 0:
            return self._failure(model, "empty_file", f"Input file is empty: {file_path}",
                                 "Provide a non-empty media file.")
        key = self._cache_key(model, {"file": self._file_hash(file_path), "options": options})
        cached = self._read_cache(key)
        if cached is not None:
            cached.cached = True
            return cached
        try:
            headers = self._headers(model)
            async with self._semaphore:
                async with self._client() as client:
                    with file_path.open("rb") as file:
                        response = await self._send_with_retries(
                            client,
                            "POST",
                            self._paths[model],
                            headers=headers,
                            files={"file": (file_path.name, file, "application/octet-stream")},
                            data=options,
                        )
            result = self._response_result(model, response, file_path.stat().st_size)
        except (httpx.HTTPError, OSError) as exc:
            result = self._failure(model, "request_failed", str(exc),
                                   "Check credentials, network access, and the input file.")
        self._write_cache(key, result)
        return result

    async def _json_request(self, model: ModelName, payload: dict[str, Any]) -> BodhanResult:
        key = self._cache_key(model, payload)
        cached = self._read_cache(key)
        if cached is not None:
            cached.cached = True
            return cached
        try:
            async with self._semaphore:
                async with self._client() as client:
                    response = await self._send_with_retries(
                        client, "POST", self._paths[model],
                        headers={**self._headers(model), "Content-Type": "application/json"},
                        json=payload,
                    )
            result = self._response_result(model, response, self._input_size(payload))
        except (httpx.HTTPError, OSError) as exc:
            result = self._failure(model, "request_failed", str(exc),
                                   "Check credentials, network access, and request parameters.")
        self._write_cache(key, result)
        return result

    async def _send_with_retries(
        self, client: httpx.AsyncClient, method: str, path: str, **kwargs: Any
    ) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            try:
                response = await client.request(method, path, **kwargs)
            except httpx.TimeoutException:
                if attempt >= self.max_retries:
                    raise
                await asyncio.sleep(2**attempt)
                continue
            if response.status_code not in {429, 500, 502, 503, 504} or attempt >= self.max_retries:
                return response
            await asyncio.sleep(2**attempt)
        raise RuntimeError("unreachable retry state")

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url, timeout=self.timeout, transport=self._transport
        )

    def _headers(self, model: ModelName) -> dict[str, str]:
        key = os.getenv(self._keys[model])
        if not key:
            raise httpx.RequestError(f"Missing environment variable {self._keys[model]}")
        return {"Authorization": f"Bearer {key}", "Accept": "application/json"}

    def _response_result(
        self, model: ModelName, response: httpx.Response, input_size: int
    ) -> BodhanResult:
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.is_error:
            message = body.get("error", response.text) if isinstance(body, dict) else response.text
            return self._failure(
                model,
                f"http_{response.status_code}",
                message,
                "Review the request fields and Bodhan API documentation.",
            )
        data = body if isinstance(body, dict) else {"value": body}
        return BodhanResult(
            ok=True, model=model, data=data, cost_inr=self._cost(model, input_size, body)
        )

    def _failure(self, model: ModelName, code: str, message: str, hint: str) -> BodhanResult:
        return BodhanResult(
            ok=False, model=model, error=BodhanError(code=code, message=message, hint=hint)
        )

    def _cost(self, model: ModelName, input_size: int, body: Any) -> float:
        if model == "transcribe":
            minutes = float(body.get("duration", 0)) / 60 if isinstance(body, dict) else 0
            return round(minutes * 0.10, 6)
        if model == "translate":
            tokens = (
                body.get("usage", {}).get("completion_tokens", 0)
                if isinstance(body, dict)
                else 0
            )
            return round(tokens * 0.20 / 10_000, 6)
        if model == "speak":
            return round(input_size * 6 / 10_000, 6)
        return 0.20 if input_size else 0.0

    def _cache_key(self, model: ModelName, payload: Any) -> str:
        canonical = json.dumps({"model": model, "payload": payload}, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _file_hash(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _read_cache(self, key: str) -> BodhanResult | None:
        path = self.cache_dir / f"{key}.json"
        if not path.is_file():
            return None
        try:
            return BodhanResult.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write_cache(self, key: str, result: BodhanResult) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / f"{key}.json").write_text(result.model_dump_json(), encoding="utf-8")

    @staticmethod
    def _input_size(payload: dict[str, Any]) -> int:
        return len(str(payload.get("text", "")).encode("utf-8"))
