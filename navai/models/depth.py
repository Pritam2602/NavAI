from __future__ import annotations

import cv2
import numpy as np
import torch


class MidasDepthEstimator:
    def __init__(self, every_n_frames: int, depth_scale: float, depth_shift: float, force_cpu: bool = False) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
        self.use_half = self.device.type == "cuda"
        self.every_n_frames = max(1, every_n_frames)
        self.depth_scale = depth_scale
        self.depth_shift = depth_shift
        self.frame_index = 0
        self.last_depth_m: np.ndarray | None = None

        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.model.to(self.device).eval()
        if self.use_half:
            self.model.half()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = transforms.small_transform

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        should_update = self.last_depth_m is None or self.frame_index % self.every_n_frames == 0
        self.frame_index += 1
        if not should_update and self.last_depth_m is not None:
            return self.last_depth_m

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb).to(self.device)
        if self.use_half:
            input_batch = input_batch.half()

        with torch.inference_mode():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_bgr.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.float().detach().cpu().numpy()
        depth_norm = cv2.normalize(depth, None, 0.0, 1.0, cv2.NORM_MINMAX)
        depth_m = self.depth_scale / (depth_norm + self.depth_shift)
        self.last_depth_m = depth_m.astype(np.float32)
        return self.last_depth_m
