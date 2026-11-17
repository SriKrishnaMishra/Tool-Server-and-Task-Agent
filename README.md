# Tool-Server-and-Task-Agent

Implementation of the Bodhan MCP tool server and task agent take-home assignment.

## Development setup

This project requires Python 3.11 or newer. Install the pinned dependencies with:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

The media tools require `ffmpeg`; PDF rendering also requires Poppler. Both are
installed by the included Dockerfile. Copy `.env.example` to `.env` and add
credentials only when running the real Bodhan integrations.

Run the checks with:

```powershell
make check
```

## MCP server

The backend runs as a standalone stdio MCP server:

```powershell
python -m bodhan_mcp
```

It exposes typed `transcribe`, `translate`, `speak`, `ocr`, and local
`detect_language` tools. Media tools take file paths rather than inline data;
Bodhan API credentials are read from the model-specific `BODHAN_*_KEY`
environment variables.

## Project structure

- `src/bodhan_mcp/` — MCP server and Bodhan model tools
- `src/bodhan_agent/` — task-agent orchestration and LLM integration
- `tests/` — automated tests
