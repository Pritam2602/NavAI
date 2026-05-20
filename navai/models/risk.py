from __future__ import annotations

from dataclasses import replace

from navai.models.fusion import Detection


HIGH_RISK_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "train",
    "chair",
    "couch",
    "bed",
    "bench",
    "dining table",
    "backpack",
    "suitcase",
    "obstacle",
    "unknown surface ahead",
    "possible stairs/drop",
}

ANIMAL_CLASSES = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}

SMALL_OBJECT_CLASSES = {
    "bottle",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "cell phone",
    "book",
    "mouse",
    "remote",
}

DEPTH_HEURISTIC_CLASSES = {"obstacle", "unknown surface ahead", "possible stairs/drop"}


def rank_navigation_risks(
    detections: list[Detection],
    danger_m: float,
    caution_m: float,
    max_distance_m: float,
) -> list[Detection]:
    assessed = []
    for detection in detections:
        if not should_keep_detection(detection, danger_m, caution_m):
            continue

        risk_score = score_detection(detection, danger_m, caution_m, max_distance_m)
        priority = priority_from_risk(detection.distance_m, risk_score, danger_m, caution_m)
        assessed.append(replace(detection, priority=priority, risk_score=risk_score))

    assessed.sort(key=lambda item: (-item.risk_score, item.distance_m))
    return assessed


def should_keep_detection(detection: Detection, danger_m: float, caution_m: float) -> bool:
    if detection.label in DEPTH_HEURISTIC_CLASSES:
        return True
    if detection.distance_m < danger_m:
        return detection.confidence >= 0.28
    if detection.label in HIGH_RISK_CLASSES:
        return detection.confidence >= 0.35
    if detection.label in ANIMAL_CLASSES:
        return detection.confidence >= 0.62 or detection.distance_m < caution_m
    if detection.label in SMALL_OBJECT_CLASSES:
        return detection.confidence >= 0.50 and detection.distance_m < caution_m
    return detection.confidence >= 0.55


def score_detection(detection: Detection, danger_m: float, caution_m: float, max_distance_m: float) -> float:
    distance_score = 45.0 * max(0.0, (max_distance_m - detection.distance_m) / max_distance_m)
    if detection.distance_m < danger_m:
        distance_score += 30.0
    elif detection.distance_m < caution_m:
        distance_score += 15.0

    direction_score = {"FRONT": 25.0, "LEFT": 10.0, "RIGHT": 10.0}.get(detection.direction, 0.0)

    if detection.label in DEPTH_HEURISTIC_CLASSES:
        class_score = 25.0
    elif detection.label in HIGH_RISK_CLASSES:
        class_score = 18.0
    elif detection.label in ANIMAL_CLASSES:
        class_score = 8.0
    elif detection.label in SMALL_OBJECT_CLASSES:
        class_score = 4.0
    else:
        class_score = 10.0

    confidence_score = 12.0 * max(0.0, min(1.0, detection.confidence))
    return round(min(100.0, distance_score + direction_score + class_score + confidence_score), 1)


def priority_from_risk(distance_m: float, risk_score: float, danger_m: float, caution_m: float) -> str:
    if distance_m < danger_m or risk_score >= 82.0:
        return "danger"
    if distance_m < caution_m or risk_score >= 58.0:
        return "caution"
    return "clear"


def select_alert_detection(detections: list[Detection]) -> Detection | None:
    for detection in detections:
        if detection.priority in {"danger", "caution"} and detection.risk_score >= 58.0:
            return detection
    return None

