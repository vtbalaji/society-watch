"""Scene handling: cut detection, optional stabilisation, automatic bin finding.

NVR exports and phone recordings of an NVR screen often switch between
cameras mid-file, so every rule and the tracker restart at a cut. Handheld
recordings also shake; with `stabilize: true` each frame is aligned to the
first frame of its scene so "did something change here" comparisons work.
"""

from __future__ import annotations

import cv2
import numpy as np


class SceneTracker:
    def __init__(self, stabilize: bool, cut_threshold: float = 0.8, max_aligned_diff: float = 28.0):
        self.stabilize = stabilize
        self.cut_threshold = cut_threshold
        self.max_aligned_diff = max_aligned_diff
        self.ref_gray = None
        self.prev_hist = None
        self.ref = None
        self.orb = cv2.ORB_create(1500) if stabilize else None
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) if stabilize else None
        self.scene_index = 0

    def update(self, frame: np.ndarray) -> tuple[bool, np.ndarray | None, np.ndarray]:
        """Returns (new_scene, affine to scene reference or None, frame in reference coords)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        cut = self.prev_hist is not None and cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL) < self.cut_threshold
        self.prev_hist = hist
        if not self.stabilize:
            if cut or self.ref is None:
                self.ref = True
                self.scene_index += cut
                return True, None, frame
            return False, None, frame

        kp, des = self.orb.detectAndCompute(gray, None)
        affine = None
        if self.ref is not None and not cut and des is not None:
            ref_kp, ref_des = self.ref
            matches = self.matcher.match(des, ref_des)
            if len(matches) > 30:
                src = np.float32([kp[m.queryIdx].pt for m in matches])
                dst = np.float32([ref_kp[m.trainIdx].pt for m in matches])
                affine, inliers = cv2.estimateAffinePartial2D(src, dst, ransacReprojThreshold=3)
                if affine is None or inliers.sum() < 25 or inliers.sum() < 0.25 * len(matches):
                    affine = None
        if affine is not None:
            # Overlays (NVR clock, logos) can align across a camera switch, so
            # also require the aligned picture to actually look like the reference.
            h, w = gray.shape
            warped = cv2.warpAffine(gray, affine, (w, h))
            valid = cv2.warpAffine(np.full_like(gray, 255), affine, (w, h)) > 0
            small = lambda im: cv2.resize(im, (w // 8, h // 8), interpolation=cv2.INTER_AREA).astype(np.float32)
            diff = np.abs(small(warped) - small(self.ref_gray))[cv2.resize(valid.astype(np.uint8), (w // 8, h // 8)) > 0]
            if diff.size == 0 or float(np.median(diff)) > self.max_aligned_diff:
                affine = None
        if affine is None:
            self.scene_index += self.ref is not None
            self.ref = (kp, des)
            self.ref_gray = gray
            return True, np.float32([[1, 0, 0], [0, 1, 0]]), frame
        h, w = frame.shape[:2]
        return False, affine, cv2.warpAffine(frame, affine, (w, h))


def warp_boxes(xyxy: np.ndarray, affine: np.ndarray | None) -> np.ndarray:
    if affine is None or not len(xyxy):
        return xyxy
    corners = np.stack([xyxy[:, [0, 1]], xyxy[:, [2, 1]], xyxy[:, [0, 3]], xyxy[:, [2, 3]]], axis=1)
    pts = corners @ affine[:, :2].T + affine[:, 2]
    return np.concatenate([pts.min(axis=1), pts.max(axis=1)], axis=1)


class BinFinder:
    """Finds rubbish bins with OWLv2 (open-vocabulary, Apache-2.0). COCO has no
    bin class, so the stock detector cannot do this."""

    MODEL = "google/owlv2-base-patch16-ensemble"
    QUERIES = ["a trash bin", "a garbage can", "a dustbin"]

    def __init__(self, threshold: float = 0.35):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.processor = Owlv2Processor.from_pretrained(self.MODEL)
        self.model = Owlv2ForObjectDetection.from_pretrained(self.MODEL).to(self.device).eval()
        self.threshold = threshold

    def __call__(self, frame_bgr: np.ndarray) -> list[tuple[list[float], float]]:
        from PIL import Image

        img = Image.fromarray(frame_bgr[:, :, ::-1])
        with self.torch.no_grad():
            inputs = self.processor(text=[self.QUERIES], images=img, return_tensors="pt").to(self.device)
            out = self.model(**inputs)
        side = max(img.size)  # OWLv2 pads to a square
        res = self.processor.post_process_grounded_object_detection(
            out, threshold=self.threshold, target_sizes=[(side, side)], text_labels=[self.QUERIES])[0]
        boxes = [(b.tolist(), float(s)) for b, s in zip(res["boxes"].cpu(), res["scores"].cpu())]
        boxes.sort(key=lambda x: -x[1])
        kept: list[tuple[list[float], float]] = []
        for b, s in boxes:  # simple NMS across the three queries
            if all(_iou(b, k[0]) < 0.3 for k in kept):
                kept.append((b, s))
        return kept


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0
