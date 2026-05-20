from __future__ import annotations

import cv2
import numpy as np

from navai.models.fusion import Detection


COLORS = {
    "danger": (40, 40, 230),
    "caution": (40, 190, 255),
    "clear": (70, 210, 80),
}


def draw_overlay(
    frame: np.ndarray,
    detections: list[Detection],
    depth_m: np.ndarray | None,
    fps: float,
    gpu_stats: dict[str, float | str],
    voice_line: str,
    show_depth: bool,
) -> np.ndarray:
    output = frame.copy()

    if show_depth and depth_m is not None:
        clipped = np.clip(depth_m, 0.0, 8.0)
        depth_vis = cv2.normalize(8.0 - clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
        output = cv2.addWeighted(output, 0.72, heatmap, 0.28, 0)

    height, width = output.shape[:2]
    cv2.drawMarker(output, (width // 2, height // 2), (255, 255, 255), cv2.MARKER_CROSS, 30, 2)

    for detection in detections:
        x1, y1, x2, y2 = (int(v) for v in detection.xyxy)
        color = COLORS[detection.priority]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.label} {detection.distance_m:.1f}m {detection.direction} R{detection.risk_score:.0f}"
        cv2.putText(output, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    hud = f"FPS {fps:04.1f} | GPU {gpu_stats.get('gpu_pct', 'n/a')}% | VRAM {gpu_stats.get('vram_gb', 'n/a')}GB | {gpu_stats.get('temp_c', 'n/a')}C"
    cv2.rectangle(output, (0, 0), (width, 36), (20, 24, 30), -1)
    cv2.putText(output, hud, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 238, 244), 2, cv2.LINE_AA)

    if voice_line:
        cv2.rectangle(output, (0, height - 42), (width, height), (20, 24, 30), -1)
        cv2.putText(output, voice_line[:110], (14, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2, cv2.LINE_AA)

    return output

