# NAVAI

NAVAI is a local real-time assistive navigation backend. It combines YOLOv8 object detection, MiDaS depth estimation, distance/direction fusion, non-blocking voice guidance, an OpenCV HUD, GPU metrics, and a WebSocket feed for the dashboard.

It also includes a depth-based unknown obstacle layer for surfaces YOLO may not classify, such as close walls, generic front obstacles, and possible stairs/drop regions. These are heuristic alerts from the depth map, not custom-trained semantic classes.

## Manual Install

```powershell
cd /d D:\VC
.\venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.\venv\Scripts\python.exe -m pip install -r requirements-app.txt
.\venv\Scripts\python.exe tools\check_env.py
```

If PyTorch reports `RuntimeError: Numpy is not available`, downgrade NumPy:

```powershell
.\venv\Scripts\python.exe -m pip install "numpy<2" --force-reinstall
```

If your environment is named `.venv_ok`, replace `.\venv\Scripts\python.exe` with `.\.venv_ok\Scripts\python.exe`.

## Run

```powershell
.\run_navai.bat
.\open_dashboard.bat
```

Controls in the OpenCV window: `q` quit, `d` depth overlay, `m` mute voice, `+` and `-` adjust alert sensitivity.

## Gemini Assistant

Create `.env` in `D:\VC`:

```text
GEMINI_API_KEY=your_api_key_here
GEMINI_MODELS=gemini-2.5-flash-lite,gemini-2.5-flash,gemini-2.5-pro
```

Then run:

```powershell
.\run_navai.bat --assistant
```

Ask questions in the terminal while the camera is running:

```text
what is ahead?
is the path clear?
what is on my left?
can I move forward?
```

NAVAI sends the latest detection/depth context to Gemini, tries the three configured models in order, speaks the answer, and pushes it to the dashboard. Gemini runs in a background worker, so camera detection keeps running while it answers.

For microphone questions, install the voice input packages and run:

```powershell
.\venv\Scripts\python.exe -m pip install SpeechRecognition PyAudio
.\run_navai.bat --assistant --voice-input
```

When the terminal says `Listening...`, ask a question aloud. The recognized question is sent to Gemini the same way typed questions are.

## Calibration

MiDaS produces relative inverse depth, not true metric depth. Tune `depth_scale` and `depth_shift` in `navai/config.py` against measured distances for your camera.
