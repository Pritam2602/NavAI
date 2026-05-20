from __future__ import annotations

import importlib.util


REQUIRED = [
    ("torch", "PyTorch inference"),
    ("torchvision", "PyTorch vision ops"),
    ("ultralytics", "YOLOv8 detector"),
    ("timm", "MiDaS dependency"),
    ("cv2", "OpenCV camera and HUD"),
    ("pyttsx3", "offline voice alerts"),
    ("numpy", "array processing"),
    ("pynvml", "GPU metrics"),
    ("websockets", "dashboard bridge"),
]


def main() -> int:
    missing = []
    print("NAVAI environment check")
    print("=======================")
    for module, purpose in REQUIRED:
        ok = importlib.util.find_spec(module) is not None
        print(f"{'OK' if ok else 'MISSING':8} {module:12} {purpose}")
        if not ok:
            missing.append(module)

    if importlib.util.find_spec("torch") is not None:
        import torch

        print()
        print(f"torch version : {torch.__version__}")
        print(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda device   : {torch.cuda.get_device_name(0)}")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

