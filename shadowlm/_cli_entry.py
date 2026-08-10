"""Console-script entry point — pure stdlib, always importable.

The real CLI (`shadowlm.cli`) is built on Typer + Rich, which the base install
ships. This shim lets the bare `shadowlm` command degrade to a friendly message
instead of a traceback if they're somehow missing (a partial or pruned install).
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from .cli import app
    except ModuleNotFoundError as e:
        if e.name in ("typer", "rich", "click", "click.core"):
            print(
                f"The shadowlm CLI needs Typer + Rich ({e.name} is missing).\n\n"
                "    pip install --upgrade --force-reinstall shadowlm\n\n"
                "They ship in the base install, so this means a partial one.",
                file=sys.stderr,
            )
            return 1
        raise
    app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
