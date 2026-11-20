"""Post-generation checks for audio, translations, and Markdown output."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Verification:
    ok: bool
    code: str = ""
    message: str = ""
    cer: float | None = None


def normalize_text(value: str) -> str:
    """Normalize text for a language-neutral character error calculation."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Return Levenshtein character error rate after Unicode/whitespace normalization."""
    expected = normalize_text(reference)
    actual = normalize_text(hypothesis)
    if not expected:
        return 0.0 if not actual else 1.0
    previous = list(range(len(actual) + 1))
    for row, left in enumerate(expected, start=1):
        current = [row]
        for column, right in enumerate(actual, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left != right),
            ))
        previous = current
    return previous[-1] / len(expected)


def normalized_cer(reference: str, hypothesis: str) -> float:
    """Compatibility name for the normalized character error rate."""
    return character_error_rate(reference, hypothesis)


def _data(result: object) -> dict[str, object]:
    if isinstance(result, dict):
        value = result.get("data", result)
    else:
        value = getattr(result, "data", {})
    return value if isinstance(value, dict) else {}


def _text(result: object, *keys: str) -> str:
    data = _data(result)
    for key in keys:
        value = data.get(key)
        if isinstance(value, str):
            return value
    return ""


def verify_audio(reference: str, transcription: object, *, max_cer: float = 0.20) -> Verification:
    actual = _text(transcription, "text", "transcript")
    if not actual:
        return Verification(False, "missing_transcript", "Back-transcription returned no text.")
    cer = character_error_rate(reference, actual)
    if cer > max_cer:
        return Verification(
            False, "audio_cer",
            f"Back-transcription CER {cer:.3f} exceeds {max_cer:.3f}.", cer
        )
    return Verification(True, cer=cer)


def _numbers(value: str) -> list[str]:
    return re.findall(r"(?<!\w)[+-]?(?:\d+(?:[.,]\d+)?%?)(?!\w)", value)


def _tables(value: str) -> list[tuple[int, int]]:
    tables: list[tuple[int, int]] = []
    lines = value.splitlines()
    index = 0
    while index < len(lines):
        if lines[index].lstrip().startswith("|"):
            block: list[str] = []
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                block.append(lines[index])
                index += 1
            tables.append((len(block), len(block[0].split("|")) - 2))
        else:
            index += 1
    return tables


def verify_translation(source: str, translation: object, *, min_length_ratio: float = 0.10,
                       max_length_ratio: float = 10.0) -> Verification:
    translated = _text(translation, "translated", "text")
    if not translated:
        return Verification(False, "missing_translation", "Translation returned no text.")
    if _numbers(source) != _numbers(translated):
        return Verification(False, "translation_numbers", "Translation changed numeric tokens.")
    source_length = len(normalize_text(source))
    ratio = len(normalize_text(translated)) / source_length if source_length else 1.0
    if not min_length_ratio <= ratio <= max_length_ratio:
        return Verification(
            False, "translation_length",
            f"Translation length ratio {ratio:.3f} is outside the allowed range."
        )
    if _tables(source) != _tables(translated):
        return Verification(False, "markdown_tables", "Translation changed Markdown table shape.")
    return Verification(True)


def verify_markdown_tables(source: str, translated: str) -> Verification:
    """Check table preservation independently for callers that already have strings."""
    if _tables(source) != _tables(translated):
        return Verification(False, "markdown_tables", "Translation changed Markdown table shape.")
    return Verification(True)
