"""Keep Windows from sleeping while a long batch runs (no power settings are changed)."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextmanager
def keep_awake() -> Iterator[bool]:
    """SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) for the duration, released on
    exit (also on errors and Ctrl-C). Yields whether the request took effect; a no-op off
    Windows. The display may still turn off; the system won't sleep."""
    if sys.platform != "win32":
        yield False
        return
    import ctypes

    k = ctypes.windll.kernel32
    ok = k.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) != 0
    try:
        yield ok
    finally:
        k.SetThreadExecutionState(ES_CONTINUOUS)
