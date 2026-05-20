from __future__ import annotations

import queue
import threading
import time

import pyttsx3


class VoiceAlertEngine:
    def __init__(self, cooldown_s: float) -> None:
        self.cooldown_s = cooldown_s
        self.enabled = True
        self.last_spoken_at: dict[str, float] = {}
        self.messages: queue.Queue[str | None] = queue.Queue()
        self.last_line = ""
        self.worker = threading.Thread(target=self._run, name="navai-tts", daemon=True)
        self.worker.start()

    def say(self, key: str, message: str, force: bool = False) -> bool:
        if not self.enabled:
            return False

        now = time.monotonic()
        if not force and now - self.last_spoken_at.get(key, 0.0) < self.cooldown_s:
            return False

        self.last_spoken_at[key] = now
        self.last_line = message
        self.messages.put(message)
        return True

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def close(self) -> None:
        self.messages.put(None)

    def _run(self) -> None:
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        while True:
            message = self.messages.get()
            if message is None:
                break
            engine.say(message)
            engine.runAndWait()

