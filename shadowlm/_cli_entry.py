"""Console-script entry point — pure stdlib, always importable.

The real CLI (`shadowlm.cli`) is built on Typer + Rich, which ship in the
`[cli]` extra (and in `[all]` / `[mlx-all]`). This shim lets the bare
`shadowlm` command degrade to a friendly message instead of a traceback when
those aren't installed.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from .cli import app
    except ModuleNotFoundError as e:
        if e.name in ("typer", "rich", "click", "click.core"):
            print(
                "The shadowlm CLI needs Typer + Rich:\n\n"
                "    pip install 'shadowlm[cli]'\n\n"
                "(included in 'shadowlm[all]' and 'shadowlm[mlx-all]')",
                file=sys.stderr,
            )
            return 1
        raise
    app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
