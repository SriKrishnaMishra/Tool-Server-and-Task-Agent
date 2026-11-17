"""Deterministic orchestration and run reporting for the Bodhan task agent."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_LANGUAGES = {
    "bengali": "bn",
    "bn": "bn",
    "english": "en",
    "en": "en",
    "gujarati": "gu",
    "gu": "gu",
    "hindi": "hi",
    "hi": "hi",
    "kannada": "kn",
    "kn": "kn",
    "malayalam": "ml",
    "ml": "ml",
    "marathi": "mr",
    "mr": "mr",
    "odia": "or",
    "oriya": "or",
    "or": "or",
    "punjabi": "pa",
    "pa": "pa",
    "tamil": "ta",
    "ta": "ta",
    "telugu": "te",
    "te": "te",
}

_UNSUPPORTED_LANGUAGE_NAMES = (
    "arabic",
    "chinese",
    "dutch",
    "french",
    "german",
    "italian",
    "japanese",
    "portuguese",
    "russian",
    "spanish",
)


@dataclass(frozen=True)
class Preflight:
    """The result of checks that do not require a model call."""

    ok: bool
    reason: str = ""
    code: str = ""
    files: tuple[Path, ...] = ()


def collect_files(files_path: Path) -> tuple[Path, ...]:
    """Return regular input files in stable order, including a single file."""
    if files_path.is_file():
        return (files_path,)
    if files_path.is_dir():
        return tuple(
            sorted((p for p in files_path.rglob("*") if p.is_file()), key=lambda p: str(p))
        )
    return ()


def preflight(task: str, files_path: Path) -> Preflight:
    """Reject missing inputs and plainly impossible requests without model calls."""
    if not files_path.exists():
        return Preflight(False, f"Input path does not exist: {files_path}", "missing_files")
    files = collect_files(files_path)
    if not files:
        return Preflight(False, f"No input files were found in: {files_path}", "missing_files")
    empty = next((path for path in files if path.stat().st_size == 0), None)
    if empty is not None:
        return Preflight(False, f"Input file is empty: {empty}", "invalid_file", files)

    lowered = task.casefold()
    if "video" in lowered and any(
        word in lowered for word in ("translate", "translation", "subtitle")
    ):
        return Preflight(
            False,
            "Video translation is not supported by the available Bodhan tools; "
            "no artifact was created.",
            "unsupported_task",
            files,
        )

    for language in _UNSUPPORTED_LANGUAGE_NAMES:
        if re.search(rf"\b{re.escape(language)}\b", lowered):
            return Preflight(
                False,
                f"The requested language ({language}) is not supported by Bodhan; "
                "no artifact was created.",
                "unsupported_language",
                files,
            )
    return Preflight(True, files=files)


class RunWriter:
    """Write the stable output layout and structured trace for one invocation."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.artifacts = output / "artifacts"
        self.output.mkdir(parents=True, exist_ok=True)
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.trace_file = output / "trace.jsonl"
        self.trace_file.unlink(missing_ok=True)
        self._step = 0

    def trace(self, kind: str, **fields: Any) -> None:
        self._step += 1
        record = {
            "timestamp": time.time(),
            "kind": kind,
            "event": kind,
            "step": self._step,
            **fields,
        }
        with self.trace_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def finish(self, result: str, report: dict[str, Any]) -> None:
        (self.output / "result.md").write_text(result.rstrip() + "\n", encoding="utf-8")
        (self.output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _failure_text(status: str, reason: str) -> str:
    return (
        f"# Bodhan task agent\n\n**Status:** {status}\n\n{reason}\n\n"
        "No fabricated artifacts were produced."
    )


def run_task(
    task: str,
    files_path: Path,
    output: Path,
    *,
    max_steps: int = 30,
    max_cost_inr: float = 20.0,
    max_seconds: float = 300.0,
) -> int:
    """Run preflight and the bounded local orchestration loop.

    Model work is intentionally not guessed: if an integration is not configured,
    the run reports that fact instead of writing synthetic translations or media.
    """
    started = time.monotonic()
    writer = RunWriter(output)
    writer.trace("plan", task=task, files=str(files_path), max_steps=max_steps,
                 max_cost_inr=max_cost_inr, max_seconds=max_seconds)

    check = preflight(task, files_path)
    if not check.ok:
        writer.trace("preflight", status="impossible", code=check.code, summary=check.reason)
        report = {
            "status": "impossible",
            "steps": 1,
            "total_cost_inr": 0.0,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "verification": [],
            "error": {"code": check.code, "message": check.reason},
        }
        writer.finish(_failure_text("impossible", check.reason), report)
        return 1

    if (
        max_steps < 2
        or max_cost_inr == 0
        or max_seconds == 0
        or time.monotonic() - started >= max_seconds
    ):
        reason = (
            "The configured budget allows no execution steps; increase the step, cost, "
            "or time budget."
        )
        writer.trace("budget", status="stopped", summary=reason, cost_inr=0.0)
        report = {
            "status": "partial",
            "steps": 1,
            "total_cost_inr": 0.0,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "verification": [],
            "error": {"code": "budget_exhausted", "message": reason},
        }
        writer.finish(_failure_text("partial", reason), report)
        return 1

    # This milestone has no implicit provider or fake fallback. A future model
    # bridge can consume this plan while retaining the same trace and report.
    reason = (
        "Preflight succeeded, but no Bodhan/LLM integration is configured for this run. "
        "The agent stopped honestly without fabricating model output."
    )
    writer.trace("tool", tool="integration", arguments={"files": len(check.files)},
                 result_summary=reason, latency_seconds=round(time.monotonic() - started, 6),
                 cost_inr=0.0)
    report = {
        "status": "failed",
        "steps": 2,
        "total_cost_inr": 0.0,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "verification": [],
        "error": {"code": "integration_unavailable", "message": reason},
    }
    writer.finish(_failure_text("failed", reason), report)
    return 1


def supported_language(value: str) -> str | None:
    """Normalize a supported language name or code, returning ``None`` otherwise."""
    return SUPPORTED_LANGUAGES.get(value.casefold().strip())
