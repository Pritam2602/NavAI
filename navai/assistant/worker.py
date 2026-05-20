from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

from navai.assistant.gemini_assistant import GeminiAssistant
from navai.models.fusion import Detection


@dataclass(frozen=True)
class AssistantJob:
    question: str
    detections: list[Detection]
    fps: float
    voice_line: str


@dataclass(frozen=True)
class AssistantResult:
    answer: str
    model: str
    ok: bool = True


class AssistantWorker:
    def __init__(self, assistant: GeminiAssistant) -> None:
        self.assistant = assistant
        self.jobs: queue.Queue[AssistantJob] = queue.Queue(maxsize=2)
        self.results: queue.Queue[AssistantResult] = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="navai-gemini-assistant", daemon=True)
        self.thread.start()

    def ask(self, job: AssistantJob) -> bool:
        try:
            self.jobs.put_nowait(job)
            return True
        except queue.Full:
            return False

    def get_result(self) -> AssistantResult | None:
        try:
            return self.results.get_nowait()
        except queue.Empty:
            return None

    def _run(self) -> None:
        while True:
            job = self.jobs.get()
            answer, model = self.assistant.answer(job.question, job.detections, job.fps, job.voice_line)
            self.results.put(AssistantResult(answer=answer, model=model, ok=model not in {"error", "local"}))
