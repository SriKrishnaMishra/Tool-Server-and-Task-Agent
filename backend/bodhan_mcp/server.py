"""MCP tool definitions for the Bodhan language models.

The tools in this module are deliberately thin adapters: validation and model
calls stay in the backend client, while the MCP layer provides stable, typed
schemas for clients.
"""

from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from .client import BodhanClient, BodhanError, BodhanResult, ModelName
from .languages import Capability, detect_script_language, normalize_language


class LanguageDetection(BaseModel):
    """Likely language and confidence derived from Unicode script characters."""

    language: str | None
    candidates: list[str]
    confidence: float = Field(ge=0, le=1)


def _language_error(model: ModelName, value: str, capability: Capability) -> BodhanResult:
    return BodhanResult(
        ok=False,
        model=model,
        error=BodhanError(
            code="unsupported_language",
            message=f"Unsupported language for {capability}: {value}",
            hint="Use one of bn, en, gu, hi, kn, ml, mr, or, pa, ta, or te.",
        ),
    )


def _normalize_or_error(
    model: ModelName, value: str, capability: Capability
) -> tuple[str | None, BodhanResult | None]:
    try:
        return normalize_language(value), None
    except ValueError:
        return None, _language_error(model, value, capability)


def create_server(client: BodhanClient | None = None) -> FastMCP:
    """Create an MCP server, optionally using an injected client for testing."""
    bodhan = client or BodhanClient()
    server = FastMCP(
        "bodhan",
        instructions=(
            "Tools for transcription, translation, speech synthesis, OCR, and "
            "script-based language detection. Media inputs are local file paths."
        ),
    )

    @server.tool(
        description=(
            "Detect likely language(s) from Unicode script characters. This is a "
            "local heuristic and does not call a model."
        )
    )
    def detect_language(text: str = Field(min_length=1, max_length=100_000)) -> LanguageDetection:
        result = detect_script_language(text)
        return LanguageDetection.model_validate(result)

    @server.tool(
        description="Transcribe a local, non-empty audio file with Bodhan Indic-Transcribe."
    )
    async def transcribe(
        audio_path: str = Field(min_length=1),
        language: str = Field(default="en", min_length=2, max_length=20),
    ) -> BodhanResult:
        code, error = _normalize_or_error("transcribe", language, "transcribe")
        if error:
            return error
        return await bodhan.transcribe(audio_path, language=code)

    @server.tool(
        description="Translate text with Bodhan Indic-Translate into a supported language."
    )
    async def translate(
        text: str = Field(min_length=1, max_length=100_000),
        source_language: str = Field(min_length=2, max_length=20),
        target_language: str = Field(min_length=2, max_length=20),
    ) -> BodhanResult:
        source, error = _normalize_or_error("translate", source_language, "translate")
        if error:
            return error
        target, error = _normalize_or_error("translate", target_language, "translate")
        if error:
            return error
        if source == target:
            return BodhanResult(
                ok=False,
                model="translate",
                error=BodhanError(
                    code="same_language",
                    message="Source and target languages must be different.",
                    hint="Choose a different target language.",
                ),
            )
        return await bodhan.translate(text, source_lang=source, target_lang=target)

    @server.tool(
        description="Synthesize speech from short text with Bodhan Indic-Speak."
    )
    async def speak(
        text: str = Field(min_length=1, max_length=10_000),
        language: str = Field(min_length=2, max_length=20),
        voice: str | None = Field(default=None, max_length=100),
    ) -> BodhanResult:
        code, error = _normalize_or_error("speak", language, "speak")
        if error:
            return error
        options: dict[str, Any] = {"language": code}
        if voice:
            options["voice"] = voice
        return await bodhan.speak(text, **options)

    @server.tool(
        description=(
            "Extract text from a local, non-empty image with Bodhan Indic-OCR. "
            "OCR currently supports English images."
        )
    )
    async def ocr(
        image_path: str = Field(min_length=1),
        language: str = Field(default="en", min_length=2, max_length=20),
    ) -> BodhanResult:
        code, error = _normalize_or_error("ocr", language, "ocr")
        if error:
            return error
        if code != "en":
            return _language_error("ocr", language, "ocr")
        return await bodhan.ocr(image_path, language=code)

    return server


mcp = create_server()


def run() -> None:
    """Run the server over MCP stdio transport."""
    mcp.run("stdio")
