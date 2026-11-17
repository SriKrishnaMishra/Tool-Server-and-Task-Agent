import pytest

from bodhan_mcp.client import BodhanResult
from bodhan_mcp.server import create_server


class FakeClient:
    async def translate(self, text: str, **options: str) -> BodhanResult:
        return BodhanResult(ok=True, model="translate", data={"text": text, **options})

    async def transcribe(self, audio_path: str, **options: str) -> BodhanResult:
        return BodhanResult(ok=True, model="transcribe", data={"path": audio_path, **options})

    async def speak(self, text: str, **options: str) -> BodhanResult:
        return BodhanResult(ok=True, model="speak", data={"text": text, **options})

    async def ocr(self, image_path: str, **options: str) -> BodhanResult:
        return BodhanResult(ok=True, model="ocr", data={"path": image_path, **options})


@pytest.mark.anyio
async def test_server_exposes_typed_initial_tools() -> None:
    server = create_server(FakeClient())
    tools = {tool.name: tool for tool in await server.list_tools()}

    assert set(tools) == {"detect_language", "transcribe", "translate", "speak", "ocr"}
    assert tools["translate"].inputSchema["properties"]["target_language"]["type"] == "string"
    assert tools["ocr"].inputSchema["properties"]["image_path"]["type"] == "string"


@pytest.mark.anyio
async def test_translate_tool_normalizes_alias_and_calls_backend() -> None:
    server = create_server(FakeClient())
    tool = server._tool_manager.get_tool("translate")

    result = await tool.fn("hello", "English", "Hindi")

    assert result.ok is True
    assert result.data["source_lang"] == "en"
    assert result.data["target_lang"] == "hi"


@pytest.mark.anyio
async def test_translate_tool_returns_actionable_validation_error() -> None:
    server = create_server(FakeClient())
    tool = server._tool_manager.get_tool("translate")

    result = await tool.fn("hello", "English", "Klingon")

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "unsupported_language"
