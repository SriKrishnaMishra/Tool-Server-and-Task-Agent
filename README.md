# Bodhan Tool Server and Task Agent

This repository provides an MCP stdio server and a bounded task agent for Bodhan
Indic speech, translation, transcription, and OCR operations.

## Setup

Requires Python 3.11 or newer.

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
.\\.venv\\Scripts\\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Fill `.env` with Bodhan model keys and, for the task agent, OpenAI-compatible
LLM settings:

- `BODHAN_TRANSCRIBE_KEY`
- `BODHAN_TRANSLATE_KEY`
- `BODHAN_SPEAK_KEY`
- `BODHAN_OCR_KEY`
- `LLM_BASE_URL`
- `LLM_MODEL`
- `LLM_API_KEY`

Do not commit `.env`. Generated media, caches, and run output are ignored.

System dependencies:

- `ffmpeg` must be on `PATH` for audio conversion and media processing.
- Poppler must be on `PATH` for scanned-PDF rendering.
- Docker is also available through the repository Dockerfile, which installs
  `ffmpeg` and Poppler.

## MCP server

Start the server over stdio:

```powershell
.\\.venv\\Scripts\\python.exe -m bodhan_mcp
```

### MCP client configuration

Add this server to VS Code with [`.vscode/mcp.json`](.vscode/mcp.json), or use
the equivalent stdio configuration in another MCP-compatible client:

```json
{
  "servers": {
    "bodhan": {
      "type": "stdio",
      "command": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
      "args": ["-m", "bodhan_mcp"],
      "env": {}
    }
  }
}
```

The server is invoked by VS Code or MCP Inspector over stdin/stdout. It reads
Bodhan credentials from `.env` and exposes the typed tools listed below.

Inspect the live server with MCP Inspector:

```powershell
npx --yes @modelcontextprotocol/inspector --cli python -m bodhan_mcp --method tools/list --format json
```

Call a local tool directly through the Inspector:

```powershell
npx --yes @modelcontextprotocol/inspector --cli python -m bodhan_mcp `
  --method tools/call `
  --tool-name detect_language `
  --tool-args-json '{\"text\":\"नमस्ते\"}'
```

No base64 or inline media is sent through MCP. Every media tool accepts a local
file path.

## Task agent

```powershell
.\\.venv\\Scripts\\python.exe -m bodhan_agent `
  --task "Translate this text to Hindi and read it aloud" `
  --files demo/inputs/hello.txt `
  --out ./out/example `
  --max-steps 30 `
  --max-cost-inr 2 `
  --max-seconds 300
```

The agent receives the task and file-path summaries, then asks the configured
LLM to plan tool calls. OCR and transcript text is treated as untrusted data.
Media is passed between tools by local path, not base64.

## MCP tools

| Tool | Purpose | Limit handling |
|---|---|---|
| `transcribe` | Bodhan Indic-Transcribe speech-to-text | Converts input to 16 kHz mono WAV and splits it into bounded windows |
| `translate` | Bodhan Indic-Translate text | Verifies translation output and preserves text/table shape |
| `speak` | Bodhan Indic-Speak short synthesis | Requires language and voice fields and writes raw WAV locally |
| `synthesize` | Chunked long TTS | Splits text at sentence boundaries, including danda, and concatenates local WAV chunks |
| `ocr` | Bodhan Indic-OCR image text extraction | Processes a local image; PDF OCR renders pages separately |
| `detect_language` | Unicode script heuristic | Used for unstated text language assumptions |
| `inspect_media` | Inspect local image/audio/video/PDF metadata | Rejects missing or unsupported inputs through structured results |
| `render_pdf_pages` | Render PDF pages locally | Processes pages one at a time |
| `list_voices` | List configured language voices | Returns the fixed voice library |
| `list_supported_languages` | List supported capabilities and codes | Keeps callers on supported language codes |

One model-specific tool per operation keeps media plumbing and limit handling
inside the server rather than requiring a third-party MCP client to implement
chunking.

## Verification and budgets

- Translations preserve verified numeric tokens and Markdown table shape.
- Every synthesis call is back-transcribed with Indic-Transcribe.
- Audio verification uses normalized character error rate; failure triggers one
  synthesis retry. Repeated failure is reported as `partial`.
- Each run is bounded by maximum steps, INR cost, wall-clock time, and per-call
  timeout.
- `trace.jsonl` records plans/tool calls, arguments, latency, and cost.
- `report.json` records status, elapsed time, cost, verification, and errors.
- `result.md` contains the final answer, assumptions, and links to artifacts.

## Recovery and limitations

The agent uses retries and structured errors for unavailable tools, malformed
responses, missing files, unsupported tasks, and exhausted budgets. It does not
invent impossible outputs.

Current demonstrated limitations:

- OCR/provider integration failed for the worksheet demo.
- The Telugu clips demo completed with local Bodhan-generated AI voice clips.
- The PDF and unsupported-French-video demo statuses are recorded in
  [demo/RESULTS.md](demo/RESULTS.md).
- The Hindi SRT demo creates a standalone `subtitles.srt` artifact from
  timestamped transcription segments when available.
- `demo/inputs/voice-note.wav` is AI-generated by Bodhan Indic-Speak, not a
  human voice recording.

## Demo results

See [demo/RESULTS.md](demo/RESULTS.md). Only statuses shown there are claimed.

## Voice samples

All fixtures below are **AI-generated by Bodhan Indic-Speak** and are not human
voice recordings. Generated WAV/MP3 media is intentionally excluded from Git;
run [demo/create_fixtures.py](demo/create_fixtures.py) to recreate every sample
after configuring the Bodhan keys.

| Local sample | Language | Purpose | Bodhan source text |
|---|---|---|---|
| `demo/inputs/voice-note.wav` | Hindi | Voice-note transcription and English SRT demo | `आज मौसम साफ़ है।` |
| `demo/inputs/clips/clip-1-te.wav` | Telugu | Telugu clip transcription | `నాకు తెలుగు వస్తుంది।` |
| `demo/inputs/clips/clip-2-hi.wav` | Hindi | Mixed-language clip transcription | `मेरा नाम आरव है।` |
| `demo/inputs/clips/clip-3-en.wav` | English | Mixed-language clip transcription | `This is a short English clip.` |
| `demo/inputs/clips/clip-4-te.wav` | Telugu | Telugu clip transcription | `ఈ రోజు చాలా మంచి రోజు.` |
| `demo/inputs/clips/clip-5-ta.wav` | Tamil | Mixed-language clip transcription | `இன்று வானிலை தெளிவாக உள்ளது।` |

Regenerate all voice fixtures:

```powershell
.\\.venv\\Scripts\\python.exe demo\create_fixtures.py
```

## Architecture

- [Architecture](docs/architecture.md)
- [Agent flow](docs/agent_flow.md)
- [PR description draft](PR_DESCRIPTION.md)

## Checks

```powershell
.\\.venv\\Scripts\\python.exe -m ruff check .
.\\.venv\\Scripts\\python.exe -m pytest
```
