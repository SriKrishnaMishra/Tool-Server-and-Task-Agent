import httpx
import pytest

from bodhan_mcp.client import BodhanClient


@pytest.mark.anyio
async def test_translate_retries_and_caches(tmp_path, monkeypatch):
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={"error": "temporary"}, request=request)
        return httpx.Response(
            200,
            json={"translated": "नमस्ते", "usage": {"completion_tokens": 10}},
            request=request,
        )

    monkeypatch.setenv("BODHAN_TRANSLATE_KEY", "test-key")
    transport = httpx.MockTransport(handler)
    async with BodhanClient(cache_dir=tmp_path, transport=transport, max_retries=1) as client:
        first = await client.translate("hello", source_lang="en", target_lang="hi")
        second = await client.translate("hello", source_lang="en", target_lang="hi")

    assert first.ok is True
    assert first.cost_inr == 0.0002
    assert second.cached is True
    assert calls == 2


@pytest.mark.anyio
async def test_missing_file_returns_structured_error(tmp_path):
    async with BodhanClient(cache_dir=tmp_path) as client:
        result = await client.ocr(tmp_path / "missing.png")

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "file_not_found"
