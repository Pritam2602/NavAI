from __future__ import annotations

import cv2
import numpy as np

from navai.models.fusion import Detection, priority_for_distance


def detect_depth_obstacles(
    depth_m: np.ndarray,
    frame_shape: tuple[int, int, int],
    existing: list[Detection],
    danger_m: float,
    caution_m: float,
) -> list[Detection]:
    height, width = depth_m.shape[:2]
    frame_h, frame_w = frame_shape[:2]
    obstacles: list[Detection] = []

    front = _front_obstacle(depth_m, frame_w, frame_h, existing, danger_m, caution_m)
    if front:
        obstacles.append(front)

    surface = _unknown_surface_ahead(depth_m, frame_w, frame_h, existing, danger_m, caution_m)
    if surface:
        obstacles.append(surface)

    drop = _possible_drop_or_stairs(depth_m, frame_w, frame_h, existing, danger_m, caution_m)
    if drop:
        obstacles.append(drop)

    return obstacles


def _front_obstacle(
    depth_m: np.ndarray,
    frame_w: int,
    frame_h: int,
    existing: list[Detection],
    danger_m: float,
    caution_m: float,
) -> Detection | None:
    if any(det.direction == "FRONT" and det.distance_m < caution_m for det in existing):
        return None

    h, w = depth_m.shape[:2]
    roi = depth_m[int(h * 0.42) : int(h * 0.82), int(w * 0.34) : int(w * 0.66)]
    if roi.size == 0:
        return None

    distance = float(np.nanpercentile(roi, 25))
    close_ratio = float(np.mean(roi < caution_m))
    if distance >= caution_m or close_ratio < 0.18:
        return None

    return Detection(
        label="obstacle",
        confidence=min(0.95, close_ratio + 0.35),
        distance_m=distance,
        direction="FRONT",
        xyxy=(frame_w * 0.34, frame_h * 0.42, frame_w * 0.66, frame_h * 0.82),
        priority=priority_for_distance(distance, danger_m, caution_m),
    )


def _unknown_surface_ahead(
    depth_m: np.ndarray,
    frame_w: int,
    frame_h: int,
    existing: list[Detection],
    danger_m: float,
    caution_m: float,
) -> Detection | None:
    if any(det.label == "unknown surface ahead" for det in existing):
        return None

    h, w = depth_m.shape[:2]
    roi = depth_m[int(h * 0.28) : int(h * 0.78), int(w * 0.22) : int(w * 0.78)]
    if roi.size == 0:
        return None

    distance = float(np.nanpercentile(roi, 35))
    close_ratio = float(np.mean(roi < caution_m))
    very_close_ratio = float(np.mean(roi < danger_m))
    texture = float(np.nanstd(roi))

    # A broad, relatively coherent close region is useful to report even if YOLO
    # cannot name it. Keep the label generic to avoid false semantic claims.
    if distance >= caution_m or close_ratio < 0.28 or texture > 1.25:
        return None
    if any(_overlaps_front(det) and det.distance_m <= distance + 0.4 for det in existing):
        return None

    return Detection(
        label="unknown surface ahead",
        confidence=min(0.92, close_ratio + very_close_ratio),
        distance_m=distance,
        direction="FRONT",
        xyxy=(frame_w * 0.22, frame_h * 0.28, frame_w * 0.78, frame_h * 0.78),
        priority=priority_for_distance(distance, danger_m, caution_m),
    )


def _overlaps_front(detection: Detection) -> bool:
    return detection.direction == "FRONT" and detection.label not in {"possible stairs/drop"}


def _possible_drop_or_stairs(
    depth_m: np.ndarray,
    frame_w: int,
    frame_h: int,
    existing: list[Detection],
    danger_m: float,
    caution_m: float,
) -> Detection | None:
    h, w = depth_m.shape[:2]
    lower = depth_m[int(h * 0.70) : int(h * 0.94), int(w * 0.28) : int(w * 0.72)]
    middle = depth_m[int(h * 0.46) : int(h * 0.66), int(w * 0.28) : int(w * 0.72)]
    if lower.size == 0 or middle.size == 0:
        return None

    lower_med = float(np.nanmedian(lower))
    middle_med = float(np.nanmedian(middle))
    edge_map = cv2.Canny(cv2.normalize(lower, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), 40, 120)
    horizontal_edges = float(np.mean(edge_map > 0))

    depth_jump = lower_med - middle_med
    if depth_jump < 1.1 and horizontal_edges < 0.08:
        return None
    if any(det.label == "possible stairs/drop" and det.distance_m <= lower_med for det in existing):
        return None

    distance = max(0.5, middle_med)
    return Detection(
        label="possible stairs/drop",
        confidence=min(0.9, 0.45 + horizontal_edges + max(0.0, depth_jump) / 5.0),
        distance_m=distance,
        direction="FRONT",
        xyxy=(frame_w * 0.28, frame_h * 0.62, frame_w * 0.72, frame_h * 0.94),
        priority=priority_for_distance(distance, danger_m, caution_m),
    )

