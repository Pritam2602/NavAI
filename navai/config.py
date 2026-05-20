from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    display_width: int = 1280
    yolo_model: str = "yolov8s.pt"
    # Keep model recall reasonably high; navai.models.risk suppresses weak/non-actionable detections later.
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    depth_every_n_frames: int = 3
    depth_scale: float = 3.5
    depth_shift: float = 0.1
    min_distance_m: float = 0.3
    max_distance_m: float = 6.0
    danger_distance_m: float = 1.2
    caution_distance_m: float = 2.5
    voice_cooldown_s: float = 2.5
    repeated_alert_cooldown_s: float = 12.0
    repeated_alert_distance_delta_m: float = 0.5
    websocket_enabled: bool = False
    websocket_host: str = "localhost"
    websocket_port: int = 8765


CONFIG = RuntimeConfig()

