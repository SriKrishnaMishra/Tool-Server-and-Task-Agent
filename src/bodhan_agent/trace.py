"""Persistent run trace and report writing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RunTrace:
    """Write one JSON object per execution event."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.artifacts = output / "artifacts"
        self.output.mkdir(parents=True, exist_ok=True)
        self.artifacts.mkdir(parents=True, exist_ok=True)
        for artifact in self.artifacts.iterdir():
            if artifact.is_file():
                artifact.unlink()
        self.path = output / "trace.jsonl"
        self.path.unlink(missing_ok=True)
        self.step = 0

    def append(
        self, tool: str, args: dict[str, Any], result_summary: str, latency: float, cost: float
    ) -> None:
        self.step += 1
        record = {
            "step": self.step,
            "kind": "plan" if tool == "preflight" else "tool",
            "tool": tool,
            "args": args,
            "result_summary": result_summary,
            "latency": round(latency, 6),
            "cost": round(cost, 6),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def finish(self, result: str, report: dict[str, Any]) -> None:
        (self.output / "result.md").write_text(result.rstrip() + "\n", encoding="utf-8")
        (self.output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
