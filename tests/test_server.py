import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

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

    assert set(tools) == {
        "detect_language",
        "transcribe",
        "translate",
        "speak",
        "synthesize",
        "ocr",
        "inspect_media",
        "render_pdf_pages",
        "list_voices",
        "list_supported_languages",
    }
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


@pytest.mark.anyio
async def test_stdio_session_calls_tools_and_returns_bad_input() -> None:
    root = Path(__file__).parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "bodhan_mcp"],
        cwd=root,
        env=env,
    )

    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            tool_map = {tool.name: tool for tool in tools.tools}

            assert "list_supported_languages" in tool_map
            assert "translate" in tool_map
            assert tool_map["translate"].description
            assert tool_map["translate"].inputSchema["properties"]["text"]["type"] == "string"

            supported = await session.call_tool("list_supported_languages")
            assert supported.isError is False

            bad_input = await session.call_tool(
                "translate",
                {
                    "text": "hello",
                    "source_language": "en",
                    "target_language": "en",
                },
            )
            assert bad_input.isError is False
            assert bad_input.structuredContent["ok"] is False
            assert bad_input.structuredContent["error"]["code"] == "same_language"
