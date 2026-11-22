"""Create small local demo inputs through the project's dependencies and Bodhan TTS."""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageDraw

from bodhan_mcp.client import BodhanClient

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "demo" / "inputs"


def create_image_and_pdf() -> None:
    worksheet = Image.new("RGB", (720, 360), "white")
    draw = ImageDraw.Draw(worksheet)
    draw.text((40, 30), "Worksheet", fill="black")
    draw.text((40, 90), "Name | Score", fill="black")
    draw.text((40, 130), "Asha | 12", fill="black")
    draw.text((40, 170), "Ravi | 15", fill="black")
    worksheet.save(INPUTS / "worksheet.png")

    pages = []
    pages_data = (
        ("Scanned Report - Page 1", ("Quarterly progress is on track.", "Three programs active.")),
        ("Scanned Report - Page 2", ("The next review is in December.", "Safety checks complete.")),
    )
    for title, lines in pages_data:
        page = Image.new("RGB", (720, 400), "white")
        page_draw = ImageDraw.Draw(page)
        page_draw.text((40, 40), title, fill="black")
        for index, line in enumerate(lines, start=1):
            page_draw.text((40, 100 + 50 * index), line, fill="black")
        pages.append(page)
    pages[0].save(INPUTS / "scanned.pdf", save_all=True, append_images=pages[1:])


async def generate_speech() -> None:
    load_dotenv(ROOT / "src" / ".env", override=False)
    load_dotenv(ROOT / ".env", override=False)
    async with BodhanClient(cache_dir=ROOT / ".cache" / "demo-fixtures") as client:
        requests = (
            ("voice-note.wav", "आज मौसम साफ़ है।", "hi"),
            ("clip-1-te.wav", "నాకు తెలుగు వస్తుంది।", "te"),
            ("clip-2-hi.wav", "मेरा नाम आरव है।", "hi"),
            ("clip-3-en.wav", "This is a short English clip.", "en"),
            ("clip-4-te.wav", "ఈ రోజు చాలా మంచి రోజు.", "te"),
            ("clip-5-ta.wav", "இன்று வானிலை தெளிவாக உள்ளது.", "ta"),
        )
        for name, text, language in requests:
            result = await client.speak(text, language=language)
            if not result.ok:
                message = result.error.message if result.error else "failed"
                print(f"{name}: Bodhan Speak unavailable: {message}")
                continue
            for key in ("audio_path", "output_file", "audio_file"):
                path = result.data.get(key)
                if isinstance(path, str) and Path(path).is_file():
                    target = (
                        INPUTS / "clips" / name
                        if name.startswith("clip-")
                        else INPUTS / name
                    )
                    target.write_bytes(Path(path).read_bytes())
                    break
            else:
                raise RuntimeError(f"{name}: Bodhan returned no local audio path")


def main() -> None:
    INPUTS.mkdir(parents=True, exist_ok=True)
    (INPUTS / "clips").mkdir(parents=True, exist_ok=True)
    (INPUTS / "hello.txt").write_text(
        "Hello, this is a short sentence for a Bengali audio demonstration.",
        encoding="utf-8",
    )
    (INPUTS / "clip.mp4").write_bytes(b"demo video placeholder")
    create_image_and_pdf()
    asyncio.run(generate_speech())


if __name__ == "__main__":
    main()
