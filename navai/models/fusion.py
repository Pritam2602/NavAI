from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from navai.models.detector import Box


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    distance_m: float
    direction: str
    xyxy: tuple[float, float, float, float]
    priority: str
    risk_score: float = 0.0


def direction_from_center(cx_norm: float) -> str:
    if cx_norm < 0.33:
        return "LEFT"
    if cx_norm < 0.66:
        return "FRONT"
    return "RIGHT"


def priority_for_distance(distance_m: float, danger_m: float, caution_m: float) -> str:
    if distance_m < danger_m:
        return "danger"
    if distance_m < caution_m:
        return "caution"
    return "clear"


def fuse_detections(
    boxes: Iterable[Box],
    depth_m: np.ndarray,
    danger_m: float,
    caution_m: float,
    min_distance_m: float = 0.3,
    max_distance_m: float = 6.0,
) -> list[Detection]:
    height, width = depth_m.shape[:2]
    detections: list[Detection] = []

    for box in boxes:
        x1n, y1n, x2n, y2n = box.xyxyn
        cx_norm = (x1n + x2n) / 2.0
        distance_m = estimate_box_distance(box, depth_m, min_distance_m, max_distance_m)

        detections.append(
            Detection(
                label=box.label,
                confidence=box.confidence,
                distance_m=distance_m,
                direction=direction_from_center(cx_norm),
                xyxy=box.xyxy,
                priority=priority_for_distance(distance_m, danger_m, caution_m),
            )
        )

    detections.sort(key=lambda item: item.distance_m)
    return detections


def estimate_box_distance(
    box: Box,
    depth_m: np.ndarray,
    min_distance_m: float,
    max_distance_m: float,
) -> float:
    height, width = depth_m.shape[:2]
    x1n, y1n, x2n, y2n = box.xyxyn
    x1 = int(np.clip(x1n * width, 0, width - 1))
    x2 = int(np.clip(x2n * width, x1 + 1, width))
    y1 = int(np.clip(y1n * height, 0, height - 1))
    y2 = int(np.clip(y2n * height, y1 + 1, height))

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    inner_x1 = x1 + int(box_w * 0.20)
    inner_x2 = x2 - int(box_w * 0.20)
    inner_y1 = y1 + int(box_h * 0.25)
    inner_y2 = y2 - int(box_h * 0.10)
    if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
        inner_x1, inner_x2, inner_y1, inner_y2 = x1, x2, y1, y2

    patch = depth_m[inner_y1:inner_y2, inner_x1:inner_x2]
    if patch.size == 0:
        cx = int(np.clip(((x1n + x2n) / 2.0) * width, 0, width - 1))
        cy = int(np.clip(((y1n + y2n) / 2.0) * height, 0, height - 1))
        raw_distance = float(depth_m[cy, cx])
    else:
        valid = patch[np.isfinite(patch)]
        raw_distance = float(np.nanpercentile(valid, 20)) if valid.size else max_distance_m

    area_ratio = ((x2n - x1n) * (y2n - y1n))
    size_cap = apparent_size_distance_cap(area_ratio)
    distance = min(raw_distance, size_cap, max_distance_m)
    return float(np.clip(distance, min_distance_m, max_distance_m))


def apparent_size_distance_cap(area_ratio: float) -> float:
    if area_ratio > 0.35:
        return 1.6
    if area_ratio > 0.22:
        return 2.2
    if area_ratio > 0.12:
        return 3.2
    if area_ratio > 0.06:
        return 4.5
    return 6.0

