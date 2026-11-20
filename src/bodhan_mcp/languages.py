"""Language aliases, script detection, and model capability checks."""

from __future__ import annotations

import re
from collections import Counter
from typing import Literal

Capability = Literal["transcribe", "translate", "speak", "ocr"]

LANGUAGES = {
    "bn": "Bengali",
    "en": "English",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
}

ALIASES = {
    **{name.lower(): code for code, name in LANGUAGES.items()},
    "bengali": "bn",
    "english": "en",
    "hindi": "hi",
    "marathi": "mr",
    "odia": "or",
    "oriya": "or",
    "punjabi": "pa",
    "tamil": "ta",
    "telugu": "te",
}

SCRIPT_RANGES = {
    "bn": r"\u0980-\u09ff",
    "gu": r"\u0a80-\u0aff",
    "pa": r"\u0a00-\u0a7f",
    "or": r"\u0b00-\u0b7f",
    "ta": r"\u0b80-\u0bff",
    "te": r"\u0c00-\u0c7f",
    "kn": r"\u0c80-\u0cff",
    "ml": r"\u0d00-\u0d7f",
    "hi": r"\u0900-\u097f",
}

SUPPORTED = {capability: set(LANGUAGES) for capability in ("transcribe", "translate", "speak")}
SUPPORTED["ocr"] = {"en"}


def normalize_language(language: str) -> str:
    """Return a canonical ISO-like language code."""
    normalized = language.strip().lower().replace("_", "-")
    code = ALIASES.get(normalized, normalized.split("-", 1)[0])
    if code not in LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    return code


def is_supported(language: str, capability: Capability) -> bool:
    """Return whether a language is supported for a model capability."""
    try:
        code = normalize_language(language)
    except ValueError:
        return False
    return code in SUPPORTED[capability]


def detect_script_language(text: str) -> dict[str, object]:
    """Detect likely languages from Unicode script characters."""
    counts: Counter[str] = Counter()
    for code, character_range in SCRIPT_RANGES.items():
        counts[code] = len(re.findall(f"[{character_range}]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if latin > max(counts.values(), default=0):
        return {"language": "en", "candidates": ["en"], "confidence": 1.0}
    best = counts.most_common()
    if not best or best[0][1] == 0:
        return {"language": None, "candidates": [], "confidence": 0.0}
    code, count = best[0]
    candidates = ["hi", "mr"] if code == "hi" else [code]
    total = sum(counts.values())
    return {
        "language": code if len(candidates) == 1 else None,
        "candidates": candidates,
        "confidence": round(count / total, 3) if total else 0.0,
    }
