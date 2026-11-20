"""Run a small, cost-bounded live smoke test against Bodhan APIs.

Usage: .venv\\Scripts\\python.exe scripts\\live_smoke.py [--image path]
If --image is omitted, a tiny English OCR test image is generated temporarily.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bodhan_mcp.client import BodhanClient, BodhanResult  # noqa: E402
from bodhan_mcp.tools.media import inspect_media  # noqa: E402

MAX_TOTAL_COST_INR = 1.0
MAX_MODEL_COST_INR = 0.5
MAX_TRANSLATION_TOKENS = 128
MAX_TRANSCRIPTION_SECONDS = 30.0
ENGLISH_SENTENCE = "The weather is clear today."
HINDI_SENTENCE = "आज मौसम साफ़ है।"


class RecordingTransport(httpx.AsyncBaseTransport):
    """Keep the raw HTTP response available when BodhanClient wraps an error."""

    def __init__(self) -> None:
        self._transport = httpx.AsyncHTTPTransport()
        self.last_status: int | None = None
        self.last_body: str | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._transport.handle_async_request(request)
        await response.aread()
        self.last_status = response.status_code
        self.last_body = response.content.decode("utf-8", errors="backslashreplace")
        return response

    async def aclose(self) -> None:
        await self._transport.aclose()


def _result_text(result: BodhanResult, *keys: str) -> str:
    for key in keys:
        value = result.data.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _print_failure(label: str, result: BodhanResult, transport: RecordingTransport) -> None:
    print(f"{label}: FAIL")
    if transport.last_status is not None:
        print(f"  HTTP status: {transport.last_status}")
        print(f"  HTTP body: {transport.last_body}")
    else:
        print("  HTTP status: unavailable (no HTTP response received)")
        message = result.error.message if result.error else "unknown"
        print(f"  HTTP body: unavailable (client error: {message})")


async def _run_speak_only() -> int:
    import os

    if not os.getenv("BODHAN_SPEAK_KEY"):
        print("Preflight: FAIL")
        print("Missing non-empty .env credential: BODHAN_SPEAK_KEY")
        return 1

    transport = RecordingTransport()
    async with BodhanClient(
        cache_dir=Path(tempfile.mkdtemp(prefix="bodhan-live-smoke-cache-")),
        timeout_seconds=60,
        max_retries=0,
        transport=transport,
    ) as live_client:
        start = time.perf_counter()
        try:
            result = await live_client.speak(HINDI_SENTENCE, language="hi", voice="Kavya")
        except Exception as exc:
            print("speak: FAIL")
            print(f"  HTTP status: {transport.last_status}")
            print(f"  HTTP body: {transport.last_body}")
            print(f"  Client error: {type(exc).__name__}: {exc}")
            return 1
        latency = time.perf_counter() - start

    raw_body = transport.last_body or ""
    print(
        f"speak: {'PASS' if result.ok and result.cost_inr < MAX_MODEL_COST_INR else 'FAIL'} "
        f"| latency={latency:.3f}s | cost=INR {result.cost_inr:.6f}"
    )
    print(f"  HTTP status: {transport.last_status}")
    print(f"  HTTP body (raw prefix): {raw_body[:256]}")
    if result.ok:
        print(f"  response_data: {result.data}")
    else:
        _print_failure("speak", result, transport)
    return int(not (result.ok and result.cost_inr < MAX_MODEL_COST_INR))


async def _run(image_arg: Path | None) -> int:
    load_dotenv(ROOT / "src" / ".env", override=False)
    load_dotenv(ROOT / ".env", override=False)
    if getattr(_run, "model", None) == "speak":
        return await _run_speak_only()
    required = (
        "BODHAN_TRANSLATE_KEY",
        "BODHAN_SPEAK_KEY",
        "BODHAN_TRANSCRIBE_KEY",
        "BODHAN_OCR_KEY",
    )
    import os

    missing = [name for name in required if not os.getenv(name)]
    if missing:
        print("Preflight: FAIL")
        print("Missing non-empty .env credentials: " + ", ".join(missing))
        return 1

    temp_dir = tempfile.TemporaryDirectory(prefix="bodhan-live-smoke-")
    temp_path = Path(temp_dir.name)
    image_path = image_arg.resolve() if image_arg else temp_path / "ocr-smoke.png"
    if image_arg is None:
        image = Image.new("RGB", (360, 90), "white")
        ImageDraw.Draw(image).text((16, 28), "BODHAN OCR TEST", fill="black")
        image.save(image_path)
    if not image_path.is_file() or image_path.stat().st_size == 0:
        print("Preflight: FAIL")
        print(f"OCR image is missing or empty: {image_path}")
        temp_dir.cleanup()
        return 1

    # The client's published accounting is Rs 0.20 per OCR image, Rs 0.10/min
    # transcription, Rs 0.20/10k translation completion tokens, and Rs 6/10k
    # UTF-8 input bytes for speech. Bound the audio and translation request so
    # the accounting ceiling stays below Rs 1 before making any paid request.
    speech_bytes = len(HINDI_SENTENCE.encode("utf-8"))
    worst_case = (
        0.20
        + MAX_TRANSCRIPTION_SECONDS / 60 * 0.10
        + MAX_TRANSLATION_TOKENS * 0.20 / 10_000
        + speech_bytes * 6 / 10_000
    )
    if worst_case >= MAX_TOTAL_COST_INR:
        print(f"Preflight: FAIL (bounded estimate INR {worst_case:.6f} is not below INR 1)")
        temp_dir.cleanup()
        return 1
    try:
        duration = float(inspect_media(image_path).get("duration") or 0)
        if duration:
            raise ValueError("OCR input unexpectedly reports a duration")
    except (OSError, ValueError) as exc:
        print(f"Preflight: FAIL (OCR image could not be inspected: {exc})")
        temp_dir.cleanup()
        return 1

    total_cost = 0.0
    ok = True
    # Use a unique cache directory so this run cannot silently substitute cached
    # responses for live Bodhan responses.
    async def call(label: str, operation: Any) -> BodhanResult | None:
        nonlocal total_cost, ok
        transport = RecordingTransport()
        async with BodhanClient(
            cache_dir=temp_path / f"cache-{label}",
            timeout_seconds=60,
            max_retries=0,
            transport=transport,
        ) as live_client:
            start = time.perf_counter()
            try:
                result = await operation(live_client)
            except Exception as exc:
                latency = time.perf_counter() - start
                print(
                    f"{label}: FAIL | latency={latency:.3f}s | cost=INR 0.000000 "
                    f"| total=INR {total_cost:.6f}"
                )
                print(f"  HTTP status: {transport.last_status}")
                print(f"  HTTP body: {transport.last_body}")
                print(f"  Client error: {type(exc).__name__}: {exc}")
                ok = False
                return None
            latency = time.perf_counter() - start
        cost = result.cost_inr
        total_cost += cost
        state = "PASS" if result.ok and total_cost < MAX_TOTAL_COST_INR else "FAIL"
        print(
            f"{label}: {state} | latency={latency:.3f}s | cost=INR {cost:.6f} "
            f"| total=INR {total_cost:.6f}"
        )
        if not result.ok:
            _print_failure(label, result, transport)
            ok = False
            return None
        if total_cost >= MAX_TOTAL_COST_INR:
            print(f"  HTTP status: {transport.last_status}")
            print(f"  HTTP body: {transport.last_body}")
            print("  Refusing further calls: reported total is not below INR 1.")
            ok = False
            return None
        return result

    translated = await call(
        "translate",
        lambda live: live.translate(
            ENGLISH_SENTENCE,
            source_lang="en",
            target_lang="hi",
            max_tokens=MAX_TRANSLATION_TOKENS,
        ),
    )
    if translated is None:
        temp_dir.cleanup()
        return 1
    translated_text = _result_text(translated, "translated", "text") or "<empty>"
    print("  translated=" + translated_text.encode("ascii", "backslashreplace").decode("ascii"))

    spoken = await call(
        "synthesize",
        lambda live: live.speak(HINDI_SENTENCE, language="hi"),
    )
    if spoken is None:
        temp_dir.cleanup()
        return 1
    audio_path_value = _result_text(spoken, "audio_path", "output_file", "audio_file")
    if not audio_path_value:
        print("synthesize: FAIL | response has no local audio path for transcription")
        print("  HTTP status: 200")
        print(f"  HTTP body: {spoken.data}")
        temp_dir.cleanup()
        return 1
    audio_path = Path(audio_path_value)
    if not audio_path.is_file():
        print(f"synthesize: FAIL | audio path does not exist: {audio_path}")
        temp_dir.cleanup()
        return 1
    audio_info = inspect_media(audio_path)
    audio_duration = float(audio_info.get("duration") or 0)
    if audio_duration <= 0 or audio_duration > MAX_TRANSCRIPTION_SECONDS:
        print(f"transcribe: FAIL | audio duration {audio_duration:.3f}s is outside (0, 30]s")
        temp_dir.cleanup()
        return 1

    transcribed = await call(
        "transcribe",
        lambda live: live.transcribe(audio_path, language="hi"),
    )
    if transcribed is None:
        temp_dir.cleanup()
        return 1
    transcription_text = _result_text(transcribed, "text", "transcript") or "<empty>"
    safe_transcription = transcription_text.encode("ascii", "backslashreplace").decode("ascii")
    print("  transcription=" + safe_transcription)

    ocr_result = await call(
        "ocr",
        lambda live: live.ocr(image_path, language="en"),
    )
    if ocr_result is None:
        temp_dir.cleanup()
        return 1
    ocr_text = _result_text(ocr_result, "text", "transcript") or "<empty>"
    print("  ocr=" + ocr_text.encode("ascii", "backslashreplace").decode("ascii"))

    temp_dir.cleanup()
    status = "PASS" if ok and total_cost < MAX_TOTAL_COST_INR else "FAIL"
    print(f"TOTAL: {status} | INR {total_cost:.6f}")
    return 0 if ok and total_cost < MAX_TOTAL_COST_INR else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Optional local image for OCR (English text).")
    parser.add_argument("--model", choices=("speak",), help="Run only one model smoke test.")
    args = parser.parse_args()
    _run.model = args.model
    return asyncio.run(_run(args.image))


if __name__ == "__main__":
    raise SystemExit(main())
