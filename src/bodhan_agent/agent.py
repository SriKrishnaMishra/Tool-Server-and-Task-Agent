"""Compatibility exports for the task agent's public API."""

from .loop import Preflight, preflight, run_task, supported_language
from .safety import check_feasibility, collect_files, data_block
from .trace import RunTrace
from .verify import (
    character_error_rate,
    normalized_cer,
    verify_audio,
    verify_markdown_tables,
    verify_translation,
)

__all__ = [
    "Preflight", "RunWriter", "collect_files", "preflight", "run_task", "supported_language",
    "check_feasibility", "data_block", "character_error_rate", "normalized_cer", "verify_audio",
    "verify_markdown_tables", "verify_translation",
]

RunWriter = RunTrace
