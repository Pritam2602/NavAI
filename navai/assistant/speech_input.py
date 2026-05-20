from __future__ import annotations

import queue
import threading


class SpeechInput:
    def __init__(self, questions: queue.Queue[str], phrase_time_limit: int = 5) -> None:
        self.questions = questions
        self.phrase_time_limit = phrase_time_limit
        self.thread = threading.Thread(target=self._run, name="navai-speech-input", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            import speech_recognition as sr
        except ImportError:
            print("Voice input needs SpeechRecognition and PyAudio.")
            print("Install with: .\\venv\\Scripts\\python.exe -m pip install SpeechRecognition PyAudio")
            return

        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True

        try:
            microphone = sr.Microphone()
        except Exception as exc:
            print(f"Could not open microphone for speech input: {exc}")
            return

        with microphone as source:
            print("Calibrating microphone noise for speech input...")
            recognizer.adjust_for_ambient_noise(source, duration=1)

        print("Voice input ready. Say a question after the prompt.")
        while True:
            try:
                with microphone as source:
                    print("Listening...")
                    audio = recognizer.listen(source, timeout=8, phrase_time_limit=self.phrase_time_limit)
                question = recognizer.recognize_google(audio).strip()
                if question:
                    print(f"You said: {question}")
                    self.questions.put(question)
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                print("I could not understand that. Please try again.")
            except sr.RequestError as exc:
                print(f"Speech recognition service error: {exc}")
            except Exception as exc:
                print(f"Speech input stopped: {exc}")
                return

