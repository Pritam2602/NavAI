from __future__ import annotations

import queue
import subprocess
import threading
import time
import os

import pyttsx3


class VoiceAlertEngine:
    def __init__(self, cooldown_s: float) -> None:
        self.cooldown_s = cooldown_s
        self.enabled = True
        self.debug = os.getenv("NAVAI_TTS_DEBUG", "0") == "1"
        self.driver = os.getenv("NAVAI_TTS_DRIVER", "powershell").strip().lower()
        self.last_spoken_at: dict[str, float] = {}
        self.last_spoken_distance: dict[str, float] = {}
        self.messages: queue.Queue[str | None] = queue.Queue()
        self.last_line = ""
        self.worker = threading.Thread(target=self._run, name="navai-tts", daemon=True)
        self.worker.start()

    def say(self, key: str, message: str, force: bool = False, distance_m: float | None = None, distance_delta_m: float = 0.0) -> bool:
        if not self.enabled:
            return False

        now = time.monotonic()
        if not force:
            elapsed = now - self.last_spoken_at.get(key, 0.0)
            previous_distance = self.last_spoken_distance.get(key)
            distance_changed = (
                distance_m is not None
                and previous_distance is not None
                and abs(distance_m - previous_distance) >= distance_delta_m
            )
            if elapsed < self.cooldown_s and not distance_changed:
                return False

        self.last_spoken_at[key] = now
        if distance_m is not None:
            self.last_spoken_distance[key] = distance_m
        self.last_line = message
        if force:
            self._clear_pending()
        self.messages.put(message)
        if self.debug:
            print(f"TTS queued: {message}")
        return True

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def close(self) -> None:
        self.messages.put(None)

    def _run(self) -> None:
        engine = None
        if self.driver == "pyttsx3":
            try:
                engine = pyttsx3.init("sapi5")
                engine.setProperty("rate", 175)
                print("TTS ready: pyttsx3/sapi5")
            except Exception as exc:
                print(f"TTS warning: pyttsx3 failed, using PowerShell speech fallback: {exc}")
        else:
            print("TTS ready: PowerShell System.Speech")

        while True:
            message = self.messages.get()
            if message is None:
                break
            try:
                if self.debug:
                    print(f"TTS speaking: {message}")
                if engine is not None:
                    engine.say(message)
                    engine.runAndWait()
                else:
                    speak_with_powershell(message)
            except Exception as exc:
                print(f"TTS warning: could not speak message: {exc}")

    def _clear_pending(self) -> None:
        while True:
            try:
                self.messages.get_nowait()
            except queue.Empty:
                return


def speak_with_powershell(message: str) -> None:
    safe_message = message.replace("'", "''")
    command = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.Rate = 0; "
        f"$s.Speak('{safe_message}')"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

