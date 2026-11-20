"""MCP tool definitions for the Bodhan language models.

The tools in this module are deliberately thin adapters: validation and model
calls stay in the backend client, while the MCP layer provides stable, typed
schemas for clients.
"""

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from .client import BodhanClient, BodhanError, BodhanResult, ModelName
from .languages import Capability, detect_script_language, normalize_language
from .tools.media import (
    concatenate_audio,
    convert_audio_to_wav,
    merge_timestamped_segments,
    split_audio_windows,
    split_translation_batches,
    split_tts_text,
)
from .tools.media import (
    inspect_media as inspect_media_file,
)
from .tools.media import (
    render_pdf_pages as render_pdf,
)


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
        try:
            wav = convert_audio_to_wav(audio_path)
            chunks = split_audio_windows(wav)
        except (FileNotFoundError, ValueError) as exc:
            return BodhanResult(
                ok=False,
                model="transcribe",
                error=BodhanError(
                    code="invalid_media", message=str(exc), hint="Provide a readable audio file."
                ),
            )
        results = []
        offsets = []
        offset = 0.0
        for chunk in chunks:
            result = await bodhan.transcribe(chunk, language=code)
            if not result.ok:
                return result
            results.append(result.data.get("segments", []))
            offsets.append(offset)
            try:
                offset += float(inspect_media_file(chunk)["duration"] or 0)
            except (FileNotFoundError, ValueError):
                pass
        if len(chunks) == 1 and not results[0]:
            return result
        data = dict(result.data)
        data["segments"] = merge_timestamped_segments(results, offsets)
        return result.model_copy(update={"data": data})

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
        batches = split_translation_batches(text)
        results = []
        for batch in batches:
            result = await bodhan.translate(batch, source_lang=source, target_lang=target)
            if not result.ok:
                return result
            results.append(str(result.data.get("translated", result.data.get("text", ""))))
        data = dict(result.data)
        data["translated"] = "\n\n".join(results)
        return result.model_copy(update={"data": data})

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
            "Synthesize local-path audio from text. Text is split at sentence boundaries, "
            "including "
            "the Indian danda, into chunks of at most 500 characters and concatenated with pauses. "
            "Returns only an output path; Bodhan Indic-Speak is used for every chunk."
        )
    )
    async def synthesize(
        text: str = Field(min_length=1, max_length=100_000),
        language: str = Field(min_length=2, max_length=20),
        voice: str | None = Field(default=None, max_length=100),
        max_cost_inr: float = Field(default=20.0, ge=0),
        output_path: str | None = Field(default=None, max_length=1_000),
    ) -> dict[str, Any]:
        code, error = _normalize_or_error("speak", language, "speak")
        if error:
            return error.model_dump()
        chunks = split_tts_text(text)
        estimate = len(text) * 6 / 10_000
        if estimate > max_cost_inr:
            return {
                "ok": False,
                "error": {
                    "code": "budget_exceeded",
                    "message": f"Estimated cost INR {estimate:.6f} exceeds the requested budget.",
                    "hint": "Increase max_cost_inr or provide shorter text.",
                },
            }
        files: list[str] = []
        total_cost = 0.0
        for chunk in chunks:
            options: dict[str, Any] = {"language": code}
            if voice:
                options["voice"] = voice
            result = await bodhan.speak(chunk, **options)
            if not result.ok:
                return result.model_dump()
            total_cost += result.cost_inr
            path = result.data.get("audio_path") or result.data.get("output_file")
            if not isinstance(path, str):
                return {
                    "ok": False,
                    "error": {
                        "code": "missing_audio_path",
                        "message": "Bodhan Speak returned no local audio path.",
                        "hint": "Configure the client to persist generated audio to a local path.",
                    },
                }
            files.append(path)
        target = output_path or str(Path.cwd() / f"bodhan-speech-{id(text)}.mp3")
        try:
            audio_file = concatenate_audio(files, target)
        except (FileNotFoundError, OSError, ValueError) as exc:
            return {
                "ok": False,
                "error": {
                    "code": "audio_concat_failed",
                    "message": str(exc),
                    "hint": "Ensure generated audio paths are readable.",
                },
            }
        return {
            "ok": True,
            "audio_file": audio_file,
            "chars": len(text),
            "cost_inr": total_cost,
            "ai_generated": True,
        }

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
        path = Path(image_path)
        if path.suffix.lower() != ".pdf":
            return await bodhan.ocr(image_path, language=code)
        try:
            pages = render_pdf(path)
        except (FileNotFoundError, OSError, ValueError) as exc:
            return {
                "ok": False,
                "error": {
                    "code": "pdf_render_failed",
                    "message": str(exc),
                    "hint": "Provide a readable PDF and install Poppler.",
                },
            }
        page_text: list[str] = []
        total_cost = 0.0
        for index, page in enumerate(pages, start=1):
            result = await bodhan.ocr(page, language=code)
            if not result.ok:
                return result
            total_cost += result.cost_inr
            page_text.append(f"## Page {index}\n\n{result.data.get('text', '')}")
        output = path.with_suffix(".ocr.md")
        try:
            output.write_text("\n\n".join(page_text) + "\n", encoding="utf-8")
        except OSError as exc:
            return {
                "ok": False,
                "error": {
                    "code": "ocr_output_failed",
                    "message": str(exc),
                    "hint": "Provide a writable directory containing the PDF.",
                },
            }
        return {
            "ok": True,
            "markdown_file": str(output),
            "pages": len(pages),
            "cost_inr": total_cost,
        }

    @server.tool(
        description="Inspect a local image, PDF, audio, or video path using Pillow and ffprobe."
    )
    def inspect_media(path: str = Field(min_length=1)) -> dict[str, Any]:
        try:
            return {"ok": True, **inspect_media_file(path)}
        except (FileNotFoundError, OSError, ValueError) as exc:
            return {
                "ok": False,
                "error": {
                    "code": "media_inspection_failed",
                    "message": str(exc),
                    "hint": "Provide a non-empty, readable media path.",
                },
            }

    @server.tool(
        description="Render a local PDF to PNG page paths. DPI must be between 72 and 600."
    )
    def render_pdf_pages(
        pdf: str = Field(min_length=1),
        dpi: int = Field(default=150, ge=72, le=600),
    ) -> dict[str, Any]:
        try:
            return {"ok": True, "pages": render_pdf(pdf, dpi)}
        except (FileNotFoundError, OSError, ValueError) as exc:
            return {
                "ok": False,
                "error": {
                    "code": "pdf_render_failed",
                    "message": str(exc),
                    "hint": "Provide a readable PDF and install Poppler.",
                },
            }

    @server.tool(description="List Bodhan Indic-Speak voices available for a language code.")
    def list_voices(language: str = Field(min_length=2, max_length=20)) -> dict[str, Any]:
        code, error = _normalize_or_error("speak", language, "speak")
        if error:
            return error.model_dump()
        return {"language": code, "voices": [{"id": f"{code}-female", "gender": "female"}]}

    @server.tool(description="List canonical language codes supported by each Bodhan capability.")
    def list_supported_languages() -> dict[str, list[str]]:
        from .languages import SUPPORTED

        return {capability: sorted(values) for capability, values in SUPPORTED.items()}

    return server


mcp = create_server()


def run() -> None:
    """Run the server over MCP stdio transport."""
    mcp.run("stdio")
