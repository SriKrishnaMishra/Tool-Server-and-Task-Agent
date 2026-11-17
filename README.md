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

## Project structure

- `backend/` — MCP server and Bodhan model tools
- `ai/` — task-agent orchestration and LLM integration
- `frontend/` — frontend scaffold for displaying task runs and traces
- `tests/` — automated tests

- [Assignment brief](./Agentic-Engineer-Take-Home-Bodhan-Tool-Server-and-Task-Agent.md)
- [Assignment email](./Assignment-Email.md)
- [Implementation prompts](./Implementation-Prompts.md)
