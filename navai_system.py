from __future__ import annotations

import argparse
import queue
import sys
import threading
from typing import Any

import cv2

from navai.assistant.gemini_assistant import DEFAULT_MODELS, GeminiAssistant
from navai.assistant.speech_input import SpeechInput
from navai.assistant.worker import AssistantJob, AssistantWorker
from navai.audio.tts import VoiceAlertEngine
from navai.config import CONFIG
from navai.models.depth import MidasDepthEstimator
from navai.models.detector import YoloDetector
from navai.models.fusion import Detection, fuse_detections
from navai.ui.overlay import draw_overlay
from navai.ui.websocket_bridge import WebSocketBridge
from navai.utils.fps_counter import FPSCounter
from navai.utils.gpu_monitor import GPUMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NAVAI real-time assistive navigation backend")
    parser.add_argument("--camera", type=int, default=CONFIG.camera_index)
    parser.add_argument("--websocket", action="store_true", help="publish live JSON to ws://localhost:8765")
    parser.add_argument("--mute", action="store_true", help="start with voice alerts muted")
    parser.add_argument("--no-depth-overlay", action="store_true", help="hide the depth heatmap at startup")
    parser.add_argument("--cpu", action="store_true", help="force CPU inference through CUDA_VISIBLE_DEVICES")
    parser.add_argument("--assistant", action="store_true", help="enable terminal questions answered by Gemini")
    parser.add_argument("--voice-input", action="store_true", help="ask Gemini questions through the microphone")
    return parser.parse_args()


def alert_text(detection: Detection, danger_m: float) -> str:
    if detection.distance_m < danger_m:
        return f"Stop. {detection.label} very close, {detection.distance_m:.1f} metres."
    if detection.direction == "FRONT":
        return f"{detection.label} ahead, {detection.distance_m:.1f} metres."
    return f"{detection.label} {detection.direction.lower()}, {detection.distance_m:.1f} metres."


def scene_status(detections: list[Detection]) -> str:
    if not detections:
        return "Path clear. No objects detected ahead."

    nearest = detections[0]
    if nearest.direction == "FRONT":
        return f"{nearest.label.capitalize()} ahead, {nearest.distance_m:.1f} metres."
    return f"Nearest object: {nearest.label} {nearest.direction.lower()}, {nearest.distance_m:.1f} metres."


def main() -> int:
    args = parse_args()

    print("Loading NAVAI models. First run may download YOLO/MiDaS weights.")
    detector = YoloDetector(CONFIG.yolo_model, CONFIG.confidence_threshold, CONFIG.iou_threshold, force_cpu=args.cpu)
    depth = MidasDepthEstimator(CONFIG.depth_every_n_frames, CONFIG.depth_scale, CONFIG.depth_shift, force_cpu=args.cpu)
    voice = VoiceAlertEngine(CONFIG.voice_cooldown_s)
    voice.set_enabled(not args.mute)
    fps_counter = FPSCounter()
    gpu_monitor = GPUMonitor()
    assistant_enabled = args.assistant or args.voice_input
    assistant = GeminiAssistant() if assistant_enabled else None
    assistant_worker = AssistantWorker(assistant) if assistant else None
    questions: queue.Queue[str] = queue.Queue()
    assistant_line = ""
    assistant_model = ""

    if assistant:
        print("Assistant models:", ", ".join(assistant.models or DEFAULT_MODELS))
        if not assistant.enabled:
            print("Assistant disabled until .env contains GEMINI_API_KEY=your_key")
        threading.Thread(target=read_questions, args=(questions,), name="navai-assistant-input", daemon=True).start()
        if args.voice_input:
            SpeechInput(questions).start()

    bridge = WebSocketBridge(CONFIG.websocket_host, CONFIG.websocket_port) if args.websocket else None
    if bridge:
        bridge.start()
        print(f"Dashboard bridge live at ws://{CONFIG.websocket_host}:{CONFIG.websocket_port}")

    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.camera_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.camera_height)
    if not capture.isOpened():
        print(f"Could not open camera index {args.camera}", file=sys.stderr)
        return 2

    show_depth = not args.no_depth_overlay
    alert_distance_m = CONFIG.caution_distance_m
    alert_count = 0

    print("Controls: q quit | d depth overlay | m mute | +/- alert sensitivity")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Camera frame read failed", file=sys.stderr)
                return 3

            boxes = detector.detect(frame)
            depth_m = depth.estimate(frame)
            detections = fuse_detections(boxes, depth_m, CONFIG.danger_distance_m, alert_distance_m)
            fps = fps_counter.tick()
            gpu_stats = gpu_monitor.read()

            nearest = detections[0] if detections else None
            if nearest and nearest.distance_m < alert_distance_m:
                if voice.say(nearest.label, alert_text(nearest, CONFIG.danger_distance_m)):
                    alert_count += 1

            while assistant_worker and not questions.empty():
                question = questions.get()
                accepted = assistant_worker.ask(AssistantJob(question, detections.copy(), fps, voice.last_line))
                if accepted:
                    print(f"NAVAI: thinking about '{question}'")
                else:
                    print("NAVAI: assistant is busy; try again in a moment.")

            if assistant_worker:
                result = assistant_worker.get_result()
                if result:
                    assistant_model = result.model
                    if result.ok:
                        assistant_line = result.answer
                        voice.say("assistant", assistant_line, force=True)
                        print(f"NAVAI ({assistant_model}): {assistant_line}")
                    else:
                        print(f"NAVAI assistant notice: {result.answer}")

            visible_line = assistant_line or voice.last_line or scene_status(detections)
            overlay = draw_overlay(frame, detections, depth_m, fps, gpu_stats, visible_line, show_depth)
            cv2.imshow("NAVAI Autonomous Navigation", overlay)

            if bridge:
                bridge.publish(payload(fps, detections, gpu_stats, visible_line, alert_count, assistant_model))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_depth = not show_depth
            if key == ord("m"):
                voice.set_enabled(not voice.enabled)
                voice.last_line = "Voice muted" if not voice.enabled else "Voice enabled"
            if key in (ord("+"), ord("=")):
                alert_distance_m = min(6.0, alert_distance_m + 0.2)
                voice.last_line = f"Alert range {alert_distance_m:.1f} metres"
            if key in (ord("-"), ord("_")):
                alert_distance_m = max(CONFIG.danger_distance_m, alert_distance_m - 0.2)
                voice.last_line = f"Alert range {alert_distance_m:.1f} metres"
    finally:
        voice.close()
        capture.release()
        cv2.destroyAllWindows()

    return 0


def payload(
    fps: float,
    detections: list[Detection],
    gpu_stats: dict[str, Any],
    voice_line: str,
    alert_count: int,
    assistant_model: str,
) -> dict[str, Any]:
    return {
        "fps": round(fps, 1),
        "detections": [d.__dict__ for d in detections],
        "det_count": len(detections),
        "nearest_m": round(detections[0].distance_m, 2) if detections else None,
        "vram_gb": gpu_stats.get("vram_gb"),
        "gpu_pct": gpu_stats.get("gpu_pct"),
        "temp_c": gpu_stats.get("temp_c"),
        "voice_line": voice_line,
        "alert_count": alert_count,
        "assistant_model": assistant_model,
    }


def read_questions(questions: queue.Queue[str]) -> None:
    print("Ask NAVAI questions in this terminal, for example: what is ahead?")
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            return
        if question:
            questions.put(question)


if __name__ == "__main__":
    raise SystemExit(main())
