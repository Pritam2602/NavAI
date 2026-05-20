from __future__ import annotations

import collections
import time


class FPSCounter:
    def __init__(self, window: int = 30) -> None:
        self.timestamps: collections.deque[float] = collections.deque(maxlen=window)

    def tick(self) -> float:
        now = time.perf_counter()
        self.timestamps.append(now)
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0

