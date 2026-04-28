"""
tictoc.py — Simple execution timer for DeepGuard / Sach-AI.
"""

import time
from datetime import datetime


class TicToc:
    """Lightweight wall-clock timer."""

    def __init__(self):
        self.start_time: float | None = None
        self.end_time:   float | None = None

    def tic(self) -> None:
        """Start (or restart) the timer."""
        self.start_time = time.time()
        print(f"[TicToc] Started at {datetime.now().strftime('%H:%M:%S')}")

    def toc(self) -> float:
        """Stop the timer and return elapsed seconds."""
        if self.start_time is None:
            return 0.0
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(
            f"[TicToc] Ended at {datetime.now().strftime('%H:%M:%S')} "
            f"— elapsed: {elapsed:.2f}s"
        )
        return elapsed

    def get_times(self) -> tuple[str, str]:
        """Return (start_str, end_str) as human-readable HH:MM:SS strings."""
        fmt  = "%H:%M:%S"
        start = (
            datetime.fromtimestamp(self.start_time).strftime(fmt)
            if self.start_time else "N/A"
        )
        end = (
            datetime.fromtimestamp(self.end_time).strftime(fmt)
            if self.end_time else "N/A"
        )
        return start, end
