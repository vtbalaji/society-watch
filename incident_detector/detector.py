"""RF-DETR wrapper that returns supervision Detections for the classes we care about."""

from __future__ import annotations

import numpy as np
import supervision as sv
from rfdetr.assets.coco_classes import COCO_CLASSES

NAME_TO_ID = {name: cid for cid, name in COCO_CLASSES.items()}

MODELS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}


def class_name(class_id: int) -> str:
    return COCO_CLASSES.get(int(class_id), str(class_id))


class Detector:
    def __init__(self, size: str = "medium", classes: list[str] | None = None, threshold: float = 0.25):
        import rfdetr

        self.name = f"rf-detr-{size}-coco"
        self.model = getattr(rfdetr, MODELS[size])()
        # Traced fp16 roughly halves latency on Apple GPUs (MPS) with the same
        # detections; fall back to plain eager fp32 if tracing fails.
        try:
            import torch

            self.model.inference(dtype=torch.float16)
        except Exception:
            pass
        self.threshold = threshold
        self.class_ids = np.array([NAME_TO_ID[c] for c in classes]) if classes else None

    def __call__(self, frame_bgr: np.ndarray) -> sv.Detections:
        dets = self.model.predict(frame_bgr[:, :, ::-1].copy(), threshold=self.threshold)
        if self.class_ids is not None and len(dets):
            dets = dets[np.isin(dets.class_id, self.class_ids)]
        return dets
