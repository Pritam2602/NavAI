from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from ultralytics import YOLO


@dataclass(frozen=True)
class Box:
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    xyxyn: tuple[float, float, float, float]


class YoloDetector:
    def __init__(self, model_name: str, confidence: float, iou: float, force_cpu: bool = False) -> None:
        self.device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        self.use_half = self.device == "cuda"
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.iou = iou

        if self.use_half:
            self.model.to(self.device)
            self.model.model.half()

    def detect(self, frame_bgr: np.ndarray) -> List[Box]:
        results = self.model.predict(
            source=frame_bgr,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            half=self.use_half,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]
        boxes: list[Box] = []
        names = result.names

        for raw_box in result.boxes:
            cls_id = int(raw_box.cls.item())
            label = names.get(cls_id, str(cls_id))
            confidence = float(raw_box.conf.item())
            xyxy = tuple(float(v) for v in raw_box.xyxy[0].tolist())
            xyxyn = tuple(float(v) for v in raw_box.xyxyn[0].tolist())
            boxes.append(Box(label=label, confidence=confidence, xyxy=xyxy, xyxyn=xyxyn))

        return boxes
