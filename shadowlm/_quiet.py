"""Swallow a backend's own stdout/stderr so shadowLM owns the console.

Backends (mlx-lm, transformers, huggingface_hub) print their own progress lines
and tqdm bars. shadowLM prints its own clean output to the real terminal
(`sys.__stdout__`, captured in models.py), which these redirects don't touch — so
we silence the backend chatter without losing our own. `SHADOWLM_DEBUG=1` shows
the raw output.
"""

from __future__ import annotations

import contextlib
import os


@contextlib.contextmanager
def quiet_backend():
    if os.environ.get("SHADOWLM_DEBUG"):
        yield
        return
    with open(os.devnull, "w") as devnull, \
            contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        yield
