from __future__ import annotations

import time


class GPUMonitor:
    def __init__(self, poll_every_s: float = 2.0) -> None:
        self.poll_every_s = poll_every_s
        self.last_poll = 0.0
        self.stats: dict[str, float | str] = {"gpu_pct": "n/a", "vram_gb": "n/a", "temp_c": "n/a"}
        self.handle = None
        try:
            import pynvml

            self.pynvml = pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.pynvml = None

    def read(self) -> dict[str, float | str]:
        now = time.monotonic()
        if now - self.last_poll < self.poll_every_s:
            return self.stats
        self.last_poll = now

        if not self.pynvml or self.handle is None:
            return self.stats

        util = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        mem = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        temp = self.pynvml.nvmlDeviceGetTemperature(self.handle, self.pynvml.NVML_TEMPERATURE_GPU)
        self.stats = {
            "gpu_pct": float(util.gpu),
            "vram_gb": round(mem.used / (1024**3), 2),
            "temp_c": float(temp),
        }
        return self.stats

