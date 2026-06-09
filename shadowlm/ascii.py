"""Startup banner for shadow training."""

from __future__ import annotations

import functools
import os

_HEART = r"""
╔══════════════════════════════════════════════════════╗
║             ♥                         ♥♥             ║
║            ♥♥♥                       ♥♥♥♥            ║
║           ♥♥♥♥♥                      ♥♥♥♥♥           ║
║          ♥♥♥♥♥♥♥                    ♥♥♥♥♥♥♥          ║
║         ♥♥♥♥♥♥♥♥                   ♥♥♥♥♥♥♥♥          ║
║         ♥♥♥♥♥♥♥♥♥                 ♥♥♥♥♥♥♥♥♥♥         ║
║         ♥♥♥♥♥♥♥♥♥♥               ♥♥♥♥♥♥♥♥♥♥♥         ║
║         ♥♥♥♥♥♥♥♥♥♥♥             ♥♥♥♥♥♥♥♥♥♥♥          ║
║          ♥♥♥♥♥♥♥♥♥♥♥           ♥♥♥♥♥♥♥♥♥♥♥♥          ║
║           ♥♥♥♥♥♥♥♥♥♥♥         ♥♥♥♥♥♥♥♥♥♥♥♥           ║
║            ♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥            ║
║             ♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥              ║
║               ♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥                ║
║                ♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥                 ║
║                  ♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥                   ║
║                    ♥♥♥♥♥♥♥♥♥♥♥♥♥                     ║
║                      ♥♥♥♥♥♥♥                         ║
║                        ♥♥                            ║
║                                                      ║
╚══════════════════════════════════════════════════════╝"""

_NAME = r"""
███████╗██╗  ██╗ █████╗ ██████╗  ██████╗ ██╗    ██╗██╗     ███╗   ███╗
██╔════╝██║  ██║██╔══██╗██╔══██╗██╔═══██╗██║    ██║██║     ████╗ ████║
███████╗███████║███████║██║  ██║██║   ██║██║ █╗ ██║██║     ██╔████╔██║
╚════██║██╔══██║██╔══██║██║  ██║██║   ██║██║███╗██║██║     ██║╚██╔╝██║
███████║██║  ██║██║  ██║██████╔╝╚██████╔╝╚███╔███╔╝███████╗██║ ╚═╝ ██║
╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚══╝╚══╝ ╚══════╝╚═╝     ╚═╝
               datasets  →  finetune  →  inference     
                 from  Lyzr  Research  Labs ·  ♥
"""

# Print the banner at most once per process, even across repeated finetunes.
_shown = False


def run_on_main_rank(fn):
    """Only run on the main process — rank 0 in a distributed (multi-GPU) launch.

    Reads the usual launcher env vars; a plain single-process run is rank 0, so
    the banner prints normally. Keeps worker ranks quiet once the studio fans
    training out across GPUs.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        rank = int(os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or 0)
        if rank == 0:
            return fn(*args, **kwargs)
        return None

    return wrapper


@run_on_main_rank
def print_ascii_art(*, once: bool = True) -> None:
    """Print the shadowLM startup banner. By default shows only once per process."""
    global _shown
    if once and _shown:
        return
    _shown = True
    print(_HEART)
    print(_NAME)
    print("Starting training session...\n")
