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
) -> list[Detection]:
    height, width = depth_m.shape[:2]
    detections: list[Detection] = []

    for box in boxes:
        x1n, y1n, x2n, y2n = box.xyxyn
        cx_norm = (x1n + x2n) / 2.0
        cy_norm = (y1n + y2n) / 2.0
        cx = int(np.clip(cx_norm * width, 0, width - 1))
        cy = int(np.clip(cy_norm * height, 0, height - 1))

        radius = 4
        patch = depth_m[max(0, cy - radius) : min(height, cy + radius + 1), max(0, cx - radius) : min(width, cx + radius + 1)]
        distance_m = float(np.nanmedian(patch)) if patch.size else float(depth_m[cy, cx])

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

