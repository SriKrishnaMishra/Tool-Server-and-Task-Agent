import math
import struct
import wave
from pathlib import Path

import pytest
from PIL import Image

from bodhan_mcp.tools.media import (
    HARD_AUDIO_CUT_SECONDS,
    MAX_AUDIO_WINDOW_SECONDS,
    inspect_media,
    merge_timestamped_segments,
    split_audio_windows,
    split_translation_batches,
    split_tts_text,
)


def test_inspect_image_reports_path_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (4, 3)).save(image_path)

    result = inspect_media(image_path)

    assert result["type"] == "image"
    assert result["width"] == 4
    assert result["height"] == 3


def test_inspect_media_rejects_missing_and_empty_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        inspect_media(tmp_path / "missing.wav")
    empty = tmp_path / "empty.wav"
    empty.touch()
    with pytest.raises(ValueError):
        inspect_media(empty)


def test_audio_windows_are_at_most_thirty_seconds_and_fallback_is_twenty_nine(
    tmp_path: Path,
) -> None:
    source = tmp_path / "long.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        frames = (
            struct.pack("<h", int(10_000 * math.sin(2 * math.pi * 440 * index / 16_000)))
            for index in range(16_000 * 31)
        )
        stream.writeframes(b"".join(frames))

    chunks = split_audio_windows(source, tmp_path / "chunks")

    assert len(chunks) >= 2
    assert all(inspect_media(chunk)["duration"] <= MAX_AUDIO_WINDOW_SECONDS for chunk in chunks)
    assert inspect_media(chunks[0])["duration"] <= HARD_AUDIO_CUT_SECONDS


def test_timestamp_offsets_are_applied_and_ordered() -> None:
    result = merge_timestamped_segments(
        [[{"start": 0.2, "end": 1.0, "text": "a"}], [{"start": 0.1, "end": 0.4, "text": "b"}]],
        [0, 29],
    )

    assert result == [
        {"start": 0.2, "end": 1.0, "text": "a"},
        {"start": 29.1, "end": 29.4, "text": "b"},
    ]


def test_tts_chunks_split_at_danda_and_respect_custom_limit() -> None:
    chunks = split_tts_text("पहला वाक्य। दूसरा वाक्य! तीसरा वाक्य?", max_chars=15)

    assert all(len(chunk) <= 15 for chunk in chunks)
    assert chunks[0].endswith("।")


def test_tts_chunks_split_oversized_words() -> None:
    chunks = split_tts_text("x" * 11, max_chars=4)

    assert chunks == ["xxxx", "xxxx", "xxx"]


def test_translation_batches_keep_markdown_table_together() -> None:
    table = "| A | B |\n|---|---|\n| 1 | 2 |"
    batches = split_translation_batches(f"Intro paragraph.\n\n{table}\n\nFinal paragraph.", 10)

    assert table in batches
    assert not any(table not in batch and "| A |" in batch for batch in batches)


def test_translation_batches_split_oversized_paragraphs() -> None:
    batches = split_translation_batches("one two three four", max_chars=7)

    assert batches == ["one two", "three", "four"]
    assert all(len(batch) <= 7 for batch in batches)


def test_media_limits_reject_invalid_values() -> None:
    with pytest.raises(ValueError):
        split_tts_text("text", 0)
    with pytest.raises(ValueError):
        split_translation_batches("text", 0)
    with pytest.raises(ValueError, match="string"):
        split_tts_text(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="string"):
        split_translation_batches(None)  # type: ignore[arg-type]


def test_pdf_render_dpi_limit_is_validated_before_rendering(tmp_path: Path) -> None:
    from bodhan_mcp.tools.media import render_pdf_pages

    pdf = tmp_path / "input.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError, match="dpi"):
        render_pdf_pages(pdf, 71)
    with pytest.raises(ValueError, match="dpi"):
        render_pdf_pages(pdf, 601)
