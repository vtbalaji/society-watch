"""Candidate rules over tracked detections.

Rules are deliberately recall-oriented: they flag clips worth a human (or a
VLM) looking at. None of them proves misconduct on its own.

- dog_dwell: a dog stays nearly still inside a zone for N seconds
  (sniffing / squatting, i.e. possible fouling).
- object_left_behind: a small object appears near a person, stays put, and
  the person moves away (possible littering or an unattended item).
- person_in_restricted_zone: a person stays inside a restricted zone.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import supervision as sv

from .config import CameraConfig, Zone
from .detector import NAME_TO_ID, class_name

PERSON = NAME_TO_ID["person"]
DOG = NAME_TO_ID["dog"]


@dataclass
class Candidate:
    rule: str
    use_case: str
    t_start: float
    t_trigger: float
    zone: str
    score: float
    bbox: list[float]
    entities: list[dict]
    explanation_codes: list[str]
    extra: dict = field(default_factory=dict)


def _bottom_center(xyxy: np.ndarray) -> tuple[float, float]:
    return float((xyxy[0] + xyxy[2]) / 2), float(xyxy[3])


def _center(xyxy: np.ndarray) -> tuple[float, float]:
    return float((xyxy[0] + xyxy[2]) / 2), float((xyxy[1] + xyxy[3]) / 2)


def _zone_of(point: tuple[float, float], zones: list[Zone]) -> Zone | None:
    for z in zones:
        if cv2.pointPolygonTest(z.polygon_px.reshape(-1, 1, 2).astype(np.int32), point, False) >= 0:
            return z
    return None


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _person_near(obj_xyxy: np.ndarray, persons: np.ndarray, ratio: float) -> bool:
    cx, cy = _center(obj_xyxy)
    for p in persons:
        px, py = _bottom_center(p)
        height = p[3] - p[1]
        # Objects sit at or below hand height, so measure to the person's box,
        # not just their feet.
        dx = max(p[0] - cx, 0, cx - p[2])
        dy = max(p[1] - cy, 0, cy - py)
        if np.hypot(dx, dy) <= ratio * height:
            return True
    return False


class DogDwellRule:
    name = "dog_dwell"

    def __init__(self, cam: CameraConfig):
        self.cfg = cam.rule(self.name)
        self.zones = cam.zones_for(self.name)
        self.history: dict[int, deque] = {}
        self.last_fired: dict[int, float] = {}

    def update(self, t: float, dets: sv.Detections) -> list[Candidate]:
        out = []
        cfg = self.cfg
        window = cfg["min_seconds"]
        seen = set()
        for xyxy, cid, conf, tid in zip(dets.xyxy, dets.class_id, dets.confidence, dets.tracker_id):
            if cid != DOG or tid is None or tid < 0 or conf < cfg["min_confidence"]:
                continue
            seen.add(tid)
            zone = _zone_of(_bottom_center(xyxy), self.zones)
            h = self.history.setdefault(tid, deque())
            if zone is None:
                h.clear()
                continue
            h.append((t, _center(xyxy), float(np.hypot(xyxy[2] - xyxy[0], xyxy[3] - xyxy[1])), float(conf), zone.name, xyxy.copy()))
            while h and t - h[0][0] > window * 2:
                h.popleft()
            span = [s for s in h if t - s[0] <= window]
            if not span or t - span[0][0] < window * 0.95:
                continue
            if any(s[4] != zone.name for s in span):
                continue
            pts = np.array([s[1] for s in span])
            motion = float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)) * 2)
            diag = float(np.median([s[2] for s in span]))
            ratio = motion / diag if diag else 1.0
            if ratio > cfg["max_motion_ratio"]:
                continue
            if t - self.last_fired.get(tid, -1e9) < cfg["cooldown_seconds"]:
                continue
            self.last_fired[tid] = t
            stillness = 1.0 - ratio / cfg["max_motion_ratio"]
            score = round(0.5 * float(np.mean([s[3] for s in span])) + 0.5 * stillness, 3)
            out.append(Candidate(
                rule=self.name, use_case="dog_fouling", t_start=span[0][0], t_trigger=t,
                zone=zone.name, score=score, bbox=xyxy.tolist(),
                entities=[{"track_id": f"local-track-{tid}", "class": "dog"}],
                explanation_codes=["dog_in_zone", "dog_stationary"],
                extra={"dwell_seconds": round(t - span[0][0], 1), "motion_ratio": round(ratio, 3)},
            ))
        for tid in list(self.history):
            if tid not in seen and self.history[tid] and t - self.history[tid][-1][0] > 3:
                del self.history[tid]
        return out


@dataclass
class _StaticObject:
    oid: int
    class_id: int
    xyxy: np.ndarray
    first_seen: float
    last_seen: float
    hits: int
    conf_sum: float
    zone: str
    background: bool
    owner_seen: bool
    last_person_near: float
    fired: bool = False


class ObjectLeftBehindRule:
    """Keeps a location-keyed memory of objects instead of relying on track IDs,
    which flicker for small, still objects."""

    name = "object_left_behind"

    def __init__(self, cam: CameraConfig):
        self.cfg = cam.rule(self.name)
        self.zones = cam.zones_for(self.name)
        self.class_ids = {NAME_TO_ID[c] for c in self.cfg["classes"]}
        self.litter_ids = {NAME_TO_ID[c] for c in self.cfg["litter_classes"]}
        self.objects: list[_StaticObject] = []
        self.next_id = 1
        self.t0: float | None = None

    def update(self, t: float, dets: sv.Detections) -> list[Candidate]:
        cfg = self.cfg
        if self.t0 is None:
            self.t0 = t
        persons = dets.xyxy[dets.class_id == PERSON] if len(dets) else np.empty((0, 4))
        for xyxy, cid, conf in zip(dets.xyxy, dets.class_id, dets.confidence):
            if cid not in self.class_ids or conf < cfg["min_confidence"]:
                continue
            zone = _zone_of(_bottom_center(xyxy), self.zones)
            if zone is None:
                continue
            match = max(self.objects, key=lambda o: _iou(o.xyxy, xyxy), default=None)
            near = _person_near(xyxy, persons, cfg["near_person_ratio"])
            if match is not None and _iou(match.xyxy, xyxy) >= 0.3:
                match.xyxy = 0.8 * match.xyxy + 0.2 * xyxy
                match.last_seen = t
                match.hits += 1
                match.conf_sum += float(conf)
                if near:
                    match.last_person_near = t
            else:
                self.objects.append(_StaticObject(
                    self.next_id, int(cid), xyxy.copy(), t, t, 1, float(conf), zone.name,
                    background=(t - self.t0) < cfg["warmup_seconds"], owner_seen=near,
                    last_person_near=t if near else -1e9,
                ))
                self.next_id += 1
        # Objects being carried move with their person, so a carried bag never
        # builds up a long static life; an object's "owner" is anyone seen next
        # to it during its first few seconds.
        out = []
        for o in self.objects:
            if t - o.first_seen <= 2.0 and _person_near(o.xyxy, persons, cfg["near_person_ratio"]):
                o.owner_seen = True
            if o.fired or o.background:
                continue
            static_for = o.last_seen - o.first_seen
            unattended_for = t - max(o.last_person_near, o.first_seen)
            seen_ratio = o.hits / max(1.0, static_for * 1.0 + 1)
            if static_for < cfg["static_seconds"] or unattended_for < cfg["unattended_seconds"]:
                continue
            if t - o.last_seen > 2.0:  # object no longer visible
                continue
            if cfg["require_owner"] and not o.owner_seen:
                continue
            o.fired = True
            litter = o.class_id in self.litter_ids
            mean_conf = o.conf_sum / o.hits
            score = round(min(1.0, 0.6 * mean_conf + 0.4 * min(1.0, unattended_for / (2 * cfg["unattended_seconds"]))), 3)
            out.append(Candidate(
                rule=self.name, use_case="littering" if litter else "unattended_item",
                t_start=o.first_seen, t_trigger=t, zone=o.zone, score=score, bbox=o.xyxy.tolist(),
                entities=[{"object_id": f"local-object-{o.oid}", "class": class_name(o.class_id)}],
                explanation_codes=["object_appeared"] + (["person_was_near"] if o.owner_seen else []) + ["object_static", "person_left"],
                extra={"static_seconds": round(static_for, 1), "unattended_seconds": round(unattended_for, 1), "seen_ratio": round(seen_ratio, 2)},
            ))
        self.objects = [o for o in self.objects if t - o.last_seen < 30.0]
        return out


class PersonInZoneRule:
    name = "person_in_restricted_zone"

    def __init__(self, cam: CameraConfig):
        self.cfg = cam.rule(self.name)
        self.zones = cam.zones_for(self.name)
        self.entered: dict[int, tuple[float, str]] = {}
        self.last_fired: dict[int, float] = {}

    def update(self, t: float, dets: sv.Detections) -> list[Candidate]:
        out = []
        cfg = self.cfg
        inside = set()
        for xyxy, cid, conf, tid in zip(dets.xyxy, dets.class_id, dets.confidence, dets.tracker_id):
            if cid != PERSON or tid is None or tid < 0 or conf < cfg["min_confidence"]:
                continue
            zone = _zone_of(_bottom_center(xyxy), self.zones)
            if zone is None:
                continue
            inside.add(tid)
            start, zname = self.entered.setdefault(tid, (t, zone.name))
            if t - start >= cfg["min_seconds"] and t - self.last_fired.get(tid, -1e9) >= cfg["cooldown_seconds"]:
                self.last_fired[tid] = t
                out.append(Candidate(
                    rule=self.name, use_case="restricted_zone", t_start=start, t_trigger=t, zone=zname,
                    score=round(float(conf), 3), bbox=xyxy.tolist(),
                    entities=[{"track_id": f"local-track-{tid}", "class": "person"}],
                    explanation_codes=["person_in_restricted_zone"],
                    extra={"dwell_seconds": round(t - start, 1)},
                ))
        for tid in list(self.entered):
            if tid not in inside:
                del self.entered[tid]
        return out


ALL_RULES = [DogDwellRule, ObjectLeftBehindRule, PersonInZoneRule]


def build_rules(cam: CameraConfig) -> list:
    return [r(cam) for r in ALL_RULES if cam.rule(r.name).get("enabled")]


def classes_needed(cam: CameraConfig) -> list[str]:
    names = {"person", "dog"}
    if cam.rule("object_left_behind").get("enabled"):
        names |= set(cam.rule("object_left_behind")["classes"])
    return sorted(names)
