import httpx
import pytest
from PIL import Image

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
async def test_nested_http_error_body_is_structured(tmp_path, monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"error": {"message": "Internal server error", "code": "500"}},
            request=request,
        )

    monkeypatch.setenv("BODHAN_SPEAK_KEY", "test-key")
    transport = httpx.MockTransport(handler)
    async with BodhanClient(
        cache_dir=tmp_path,
        transport=transport,
        max_retries=0,
    ) as client:
        result = await client.speak("hello", language="en")

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "http_500"
    assert result.error.message == "Internal server error"


@pytest.mark.anyio
async def test_speak_sends_documented_request_and_reads_wav(tmp_path, monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/audio/speech"
        assert request.headers["accept"] == "audio/wav"
        body = __import__("json").loads(request.content)
        assert body == {
            "model": "indic-speak",
            "input": "hello",
            "voice": "Kavya",
            "instructions": '{"lang": "hi", "style": "news"}',
        }
        return httpx.Response(
            200,
            content=b"RIFF-wav-data",
            headers={"content-type": "audio/wav"},
            request=request,
        )

    monkeypatch.setenv("BODHAN_SPEAK_KEY", "test-key")
    async with BodhanClient(
        cache_dir=tmp_path,
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        result = await client.speak("hello", language="hi", style="news")

    assert result.ok is True
    assert result.data["audio_path"].endswith(".wav")


@pytest.mark.anyio
async def test_translate_sends_working_openai_payload(tmp_path, monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = __import__("json").loads(request.content)
        assert body == {
            "model": "indic-translate",
            "messages": [
                {
                    "role": "user",
                    "content": "Translate the following text to the language code bn:\nhello",
                }
            ],
            "source_lang": "en",
            "target_lang": "bn",
        }
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "হ্যালো"}}],
                "usage": {"completion_tokens": 1},
            },
            request=request,
        )

    monkeypatch.setenv("BODHAN_TRANSLATE_KEY", "test-key")
    async with BodhanClient(
        cache_dir=tmp_path,
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        result = await client.translate("hello", source_lang="en", target_lang="bn")

    assert result.ok is True
    assert result.data["translated"] == "হ্যালো"


@pytest.mark.anyio
async def test_file_calls_include_model_in_multipart_data(tmp_path, monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        assert b'name="model"' in request.content
        assert b"indic-ocr" in request.content
        return httpx.Response(200, json={"text": "Hello"}, request=request)

    image = tmp_path / "test.png"
    Image.new("RGB", (2, 2)).save(image)
    monkeypatch.setenv("BODHAN_OCR_KEY", "test-key")
    async with BodhanClient(
        cache_dir=tmp_path,
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        result = await client.ocr(image, language="en")

    assert result.ok is True


@pytest.mark.anyio
async def test_missing_file_returns_structured_error(tmp_path):
    async with BodhanClient(cache_dir=tmp_path) as client:
        result = await client.ocr(tmp_path / "missing.png")

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "file_not_found"


@pytest.mark.anyio
async def test_ocr_provider_error_is_actionable(tmp_path, monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={
                "error": {
                    "message": "OCR is not supported for provider: openai.",
                }
            },
            request=request,
        )

    image = tmp_path / "test.png"
    Image.new("RGB", (2, 2)).save(image)
    monkeypatch.setenv("BODHAN_OCR_KEY", "test-key")
    async with BodhanClient(
        cache_dir=tmp_path,
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        result = await client.ocr(image, language="en")

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "provider_unsupported"
    assert "Indic-OCR" in result.error.hint


@pytest.mark.anyio
async def test_requests_send_configured_api_key(tmp_path, monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(200, json={"translated": "hello"}, request=request)

    monkeypatch.setenv("BODHAN_TRANSLATE_KEY", "test-key")
    async with BodhanClient(
        cache_dir=tmp_path,
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.translate("नमस्ते", source_lang="hi", target_lang="en")

    assert result.ok is True
