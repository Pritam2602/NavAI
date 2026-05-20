from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

from navai.models.fusion import Detection


DEFAULT_MODELS = ("gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro")


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class GeminiAssistant:
    def __init__(self, api_key: str | None = None, models: tuple[str, ...] | None = None) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        configured_models = os.getenv("GEMINI_MODELS", "")
        if models is not None:
            self.models = models
        elif configured_models:
            self.models = tuple(model.strip() for model in configured_models.split(",") if model.strip())
        else:
            self.models = DEFAULT_MODELS

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def answer(self, question: str, detections: list[Detection], fps: float, voice_line: str) -> tuple[str, str]:
        if not self.enabled:
            return "Gemini is not configured. Add GEMINI_API_KEY to .env.", "local"

        prompt = self._prompt(question, detections, fps, voice_line)
        last_error = ""
        for model in self.models:
            try:
                return self._generate(model, prompt), model
            except Exception as exc:
                last_error = str(exc)

        if "HTTP 429" in last_error:
            return "Gemini is rate-limited right now. Please wait and ask again.", "error"
        return f"I could not reach Gemini right now. Last error: {last_error}", "error"

    def _prompt(self, question: str, detections: list[Detection], fps: float, voice_line: str) -> str:
        scene = {
            "fps": round(fps, 1),
            "latest_voice_alert": voice_line,
            "detections_nearest_first": [asdict(detection) for detection in detections[:12]],
        }
        return (
            "You are NAVAI, an assistive navigation voice assistant for a blind or low-vision user. "
            "Answer using only the live scene JSON. Be brief, practical, and safety-focused. "
            "Mention uncertainty when depth may be approximate. Do not invent objects. "
            "If the path is unsafe, say that first. Keep the answer under 35 words.\n\n"
            f"Live scene JSON:\n{json.dumps(scene, indent=2)}\n\n"
            f"User question: {question}"
        )

    def _generate(self, model: str, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        body: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 120},
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=12) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{model} HTTP {exc.code}: {detail[:240]}") from exc

        candidates = payload.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"{model} returned no candidates")

        parts = candidates[0].get("content", {}).get("parts", [])
        text = " ".join(part.get("text", "") for part in parts).strip()
        if not text:
            raise RuntimeError(f"{model} returned an empty answer")
        return text
