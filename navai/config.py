from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    display_width: int = 1280
    yolo_model: str = "yolov8n.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    depth_every_n_frames: int = 3
    depth_scale: float = 3.5
    depth_shift: float = 0.1
    danger_distance_m: float = 1.2
    caution_distance_m: float = 2.5
    voice_cooldown_s: float = 2.5
    websocket_enabled: bool = False
    websocket_host: str = "localhost"
    websocket_port: int = 8765


CONFIG = RuntimeConfig()

