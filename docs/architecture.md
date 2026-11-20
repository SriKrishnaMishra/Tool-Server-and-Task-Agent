# Architecture

```mermaid
flowchart LR
    USER["User / MCP client"] --> ENTRY["MCP stdio or task-agent CLI"]
    ENTRY --> MCP["bodhan_mcp FastMCP server"]
    ENTRY --> AGENT["Bodhan task agent"]
    AGENT --> PLAN["LLM planning"]
    PLAN --> LOOP["Plan / act / verify / retry"]
    LOOP --> MCP
    MCP --> SERVERTOOLS["Typed media and model tools"]
    SERVERTOOLS --> CLIENT["BodhanClient"]
    CLIENT --> BODHAN["Hosted Bodhan APIs"]
    BODHAN --> CLIENT
    CLIENT --> SERVERTOOLS
    SERVERTOOLS --> LOOP
    LOOP --> TRACE["trace.jsonl"]
    LOOP --> REPORT["report.json"]
    LOOP --> RESULT["result.md"]
    LOOP --> ARTIFACTS["artifacts/"]
```

The MCP server owns Bodhan credentials and API transport, while the agent owns
planning, budgets, safety, verification, traceability, and output writing.
Media stays in local files and is passed by path.

The implementation exposes separate tools for transcription, translation,
speech synthesis, OCR, media inspection, PDF rendering, language detection,
voice listing, and language listing. This keeps model limits in code while
remaining directly usable by an MCP client.
