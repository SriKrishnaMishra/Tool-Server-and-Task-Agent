"""Run the Bodhan MCP server."""

from .server import run


def main() -> None:
    """Start the Bodhan MCP server over stdio."""
    run()


if __name__ == "__main__":
    main()
