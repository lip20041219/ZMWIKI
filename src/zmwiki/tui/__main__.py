"""Entry point for ``python -m zmwiki.tui``."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    try:
        from .app import main as run
    except ModuleNotFoundError as exc:
        if exc.name == "textual" or (exc.name or "").startswith("textual."):
            print(
                "error: Textual is required for the TUI. "
                'Install it with: pip install -e ".[tui]"',
                file=sys.stderr,
            )
            return 2
        raise
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
