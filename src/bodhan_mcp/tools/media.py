"""Path-based media inspection, conversion, chunking, and text batching."""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

MAX_AUDIO_WINDOW_SECONDS = 30
HARD_AUDIO_CUT_SECONDS = 29
DEFAULT_TTS_CHUNK_CHARS = 500
DEFAULT_TRANSLATION_BATCH_CHARS = 4_000


def _require_file(path: str | Path) -> Path:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {file_path}")
    if file_path.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {file_path}")
    return file_path


def inspect_media(path: str | Path) -> dict[str, Any]:
    """Inspect an image, PDF, or audio/video file using local media tools."""
    file_path = _require_file(path)
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        try:
            import fitz

            with fitz.open(file_path) as document:
                pages = document.page_count
        except (ImportError, OSError, RuntimeError) as exc:
            raise ValueError(f"Could not inspect PDF file: {file_path}") from exc
        return {
            "type": "pdf",
            "duration": None,
            "size": file_path.stat().st_size,
            "pages": int(pages),
            "sample_rate": None,
        }
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}:
        with Image.open(file_path) as image:
            return {
                "type": "image",
                "duration": None,
                "size": file_path.stat().st_size,
                "pages": None,
                "sample_rate": None,
                "width": image.width,
                "height": image.height,
                "format": image.format,
            }

    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-show_streams",
                "-of",
                "json",
                str(file_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        probe = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not inspect media file: {file_path}") from exc
    streams = probe.get("streams", [])
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    return {
        "type": "audio" if audio else "video",
        "duration": float(probe.get("format", {}).get("duration", 0) or 0),
        "size": file_path.stat().st_size,
        "pages": None,
        "sample_rate": int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
    }


def render_pdf_pages(pdf: str | Path, dpi: int = 150) -> list[str]:
    """Render every PDF page to PNG files in a temporary directory."""
    file_path = _require_file(pdf)
    if file_path.suffix.lower() != ".pdf":
        raise ValueError("render_pdf_pages requires a PDF path")
    if not 72 <= dpi <= 600:
        raise ValueError("dpi must be between 72 and 600")
    output_dir = Path(tempfile.mkdtemp(prefix="bodhan-pdf-"))
    try:
        try:
            import fitz

            scale = dpi / 72
            with fitz.open(file_path) as document:
                paths = []
                for index, page in enumerate(document, start=1):
                    matrix = fitz.Matrix(scale, scale)
                    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                    output = output_dir / f"page-{index:04d}.png"
                    pixmap.save(output)
                    paths.append(str(output))
        except (ImportError, OSError, RuntimeError) as exc:
            raise ValueError(f"Could not render PDF file: {file_path}") from exc
        return paths
    except (OSError, ValueError, RuntimeError):
        for child in output_dir.iterdir():
            child.unlink(missing_ok=True)
        output_dir.rmdir()
        raise


def convert_audio_to_wav(audio: str | Path, output: str | Path | None = None) -> str:
    """Convert audio to a 16 kHz, mono WAV using ffmpeg."""
    source = _require_file(audio)
    if output:
        target = Path(output)
    else:
        temporary = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temporary.close()
        target = Path(temporary.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(source), "-ar", "16000", "-ac", "1", str(target)
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"Could not convert audio to 16 kHz mono WAV: {source}") from exc
    return str(target)


def split_audio_windows(
    audio: str | Path, output_dir: str | Path | None = None
) -> list[str]:
    """Split audio into windows no longer than 30 seconds, preferring silence."""
    source = _require_file(audio)
    duration = _probe_duration(source)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Audio duration must be a positive finite number: {source}")
    if duration <= MAX_AUDIO_WINDOW_SECONDS:
        return [str(source)]
    destination = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="bodhan-audio-"))
    destination.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    start = 0.0
    while start < duration:
        end = min(start + MAX_AUDIO_WINDOW_SECONDS, duration)
        if end < duration:
            silence = _silence_before(source, start, end)
            end = silence if silence is not None else min(start + HARD_AUDIO_CUT_SECONDS, duration)
        chunk_path = destination / f"chunk-{len(chunks):04d}.wav"
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(source),
                    "-t", f"{end - start:.3f}", "-ar", "16000", "-ac", "1", str(chunk_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(f"Could not split audio file: {source}") from exc
        chunks.append(str(chunk_path))
        start = end
    return chunks


def _probe_duration(path: Path) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            check=True, capture_output=True, text=True,
        )
        duration = float(result.stdout.strip())
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Audio duration must be a positive finite number: {path}")
        return duration
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise ValueError(f"Could not read audio duration: {path}") from exc


def _silence_before(path: Path, start: float, end: float) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(path),
                "-af", "silencedetect=noise=-35dB:d=0.25", "-f", "null", "-",
            ],
            check=False, capture_output=True, text=True,
        )
    except OSError:
        return None
    matches = re.findall(r"silence_end: ([0-9.]+)", result.stderr)
    if not matches:
        return None
    candidate = start + float(matches[-1])
    return candidate if start < candidate <= end else None


def merge_timestamped_segments(
    segments: Iterable[dict[str, Any]], offsets: Iterable[float]
) -> list[dict[str, Any]]:
    """Add each chunk offset to its transcript segments and preserve source order."""
    merged: list[dict[str, Any]] = []
    for chunk_segments, offset in zip(segments, offsets):
        for segment in chunk_segments:
            item = dict(segment)
            item["start"] = round(float(item["start"]) + offset, 3)
            item["end"] = round(float(item["end"]) + offset, 3)
            merged.append(item)
    return sorted(merged, key=lambda item: (item["start"], item["end"]))


def split_tts_text(text: str, max_chars: int = DEFAULT_TTS_CHUNK_CHARS) -> list[str]:
    """Split text at sentence boundaries, including the Indian danda, under a limit."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    sentences = [
        part.strip() for part in re.split(r"(?<=[.!?।॥])\s+", text.strip()) if part.strip()
    ]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            words = sentence.split()
            for word in words:
                if len(word) > max_chars:
                    if current:
                        chunks.append(current)
                        current = ""
                    chunks.extend(
                        word[index:index + max_chars]
                        for index in range(0, len(word), max_chars)
                    )
                    continue
                if current and len(current) + len(word) + 1 > max_chars:
                    chunks.append(current)
                    current = ""
                current = f"{current} {word}".strip()
            continue
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def split_translation_batches(
    text: str, max_chars: int = DEFAULT_TRANSLATION_BATCH_CHARS
) -> list[str]:
    """Batch paragraphs without splitting Markdown table blocks."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    units = [unit.strip() for unit in re.split(r"\n\s*\n", text) if unit.strip()]
    batches: list[str] = []
    current = ""
    for unit in units:
        is_table = any(line.lstrip().startswith("|") for line in unit.splitlines())
        if current and len(current) + len(unit) + 2 > max_chars:
            batches.append(current)
            current = ""
        if is_table and len(unit) > max_chars:
            if current:
                batches.append(current)
                current = ""
            batches.append(unit)
        elif len(unit) > max_chars:
            words = unit.split()
            partial = ""
            for word in words:
                if len(word) > max_chars:
                    if partial:
                        batches.append(partial)
                        partial = ""
                    batches.extend(
                        word[index:index + max_chars]
                        for index in range(0, len(word), max_chars)
                    )
                elif partial and len(partial) + len(word) + 1 > max_chars:
                    batches.append(partial)
                    partial = word
                else:
                    partial = f"{partial} {word}".strip()
            if partial:
                batches.append(partial)
        elif current:
            current = f"{current}\n\n{unit}"
        else:
            current = unit
    if current:
        batches.append(current)
    return batches


def concatenate_audio(files: Iterable[str | Path], output: str | Path) -> str:
    """Concatenate generated audio files with a short pause between chunks."""
    paths = [Path(file) for file in files]
    if not paths:
        raise ValueError("At least one audio file is required")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    temporary.close()
    concat_list = Path(temporary.name)
    try:
        concat_list.write_text(
            "".join(f"file '{path.resolve()}'\n" for path in paths), encoding="utf-8"
        )
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
             "-ar", "16000", "-ac", "1", str(target)],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Could not concatenate generated audio files") from exc
    finally:
        concat_list.unlink(missing_ok=True)
    return str(target)
