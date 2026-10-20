"""Candidate rules over tracked detections.

Rules are deliberately recall-oriented: they flag clips worth a human (or a
VLM) looking at. None of them proves misconduct on its own.

- dog_fouling: a dog stays on one spot (squatting) and then it and its owner
  leave without anyone bending down to clean up.
- object_left_behind: a small object appears near a person, stays put, and
  the person moves away (possible littering or an unattended item).
- person_in_restricted_zone: a person stays inside a restricted zone.
- bin_dumping: a person visits the area around a rubbish bin and, after they
  leave, something new is left on the floor beside the bin. Works for items
  the detector has no class for (garbage bags, cartons, loose waste).
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
VEHICLES = {NAME_TO_ID[c] for c in ("car", "truck", "bus", "motorcycle", "bicycle")}


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
    moment_frame: np.ndarray | None = None  # extra keyframe, e.g. the dog mid-squat
    moment_t: float | None = None


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


@dataclass
class _DogSpot:
    center: np.ndarray
    diag: float
    first_seen: float
    last_seen: float
    zone: str
    hits: int = 1
    conf_sum: float = 0.0
    bbox: list | None = None
    suspected: bool = False  # dog stayed long enough: possible fouling
    squat_frame: np.ndarray | None = None
    squat_t: float | None = None
    min_height_ratio: float = 1.0  # lowest (height / standing height) of anyone near
    people_last_near: float = -1e9
    done: bool = False


class DogFoulingRule:
    """A dog stays on one spot (squatting), then the dog and the people with it
    leave. If nobody near the spot bent down in between, flag it as fouling
    that was not cleaned up.

    Spots are remembered by location rather than tracker ID, because the
    owner often stands between the dog and the camera and the ID switches.
    """

    name = "dog_fouling"

    def __init__(self, cam: CameraConfig):
        self.cfg = cam.rule(self.name)
        self.zones = cam.zones_for(self.name)
        self.spots: list[_DogSpot] = []
        self.person_height: dict[int, float] = {}  # tallest seen per person track
        self.sample_fps = cam.settings["sample_fps"]

    def update(self, t: float, dets: sv.Detections, ctx: dict | None = None) -> list[Candidate]:
        cfg = self.cfg
        dogs = [(xyxy, float(c)) for xyxy, cid, c in zip(dets.xyxy, dets.class_id, dets.confidence)
                if cid == DOG and c >= cfg["min_confidence"]]
        people = [(xyxy, int(tid) if tid is not None else -1) for xyxy, cid, tid in zip(dets.xyxy, dets.class_id, dets.tracker_id) if cid == PERSON]
        for xyxy, tid in people:
            if tid >= 0:
                self.person_height[tid] = max(self.person_height.get(tid, 0.0), float(xyxy[3] - xyxy[1]))

        frame_w = ctx["frame"].shape[1] if ctx and ctx.get("frame") is not None else None
        for xyxy, conf in dogs:
            if frame_w is not None and (xyxy[0] < 4 or xyxy[2] > frame_w - 4):
                continue  # half out of view: position and posture are unreliable
            zone = _zone_of(_bottom_center(xyxy), self.zones)
            if zone is None:
                continue
            c = np.array(_center(xyxy))
            diag = float(np.hypot(xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]))
            spot = min((sp for sp in self.spots if not sp.done), key=lambda sp: np.linalg.norm(sp.center - c), default=None)
            if spot is not None and np.linalg.norm(spot.center - c) <= cfg["spot_radius"] * spot.diag \
                    and t - spot.last_seen <= cfg["max_gap_seconds"]:
                spot.last_seen, spot.hits, spot.conf_sum = t, spot.hits + 1, spot.conf_sum + conf
                spot.bbox = xyxy.tolist()
            else:
                self.spots.append(_DogSpot(c, diag, t, t, zone.name, 1, conf, xyxy.tolist()))

        out = []
        for sp in self.spots:
            if sp.done:
                continue
            if not sp.suspected:
                dwell = sp.last_seen - sp.first_seen
                # Dogs walking past the same spot one after another add only a
                # few sightings; a squatting dog is seen there most of the time.
                if dwell >= cfg["min_seconds"] and sp.hits >= cfg["min_seen_ratio"] * dwell * self.sample_fps:
                    sp.suspected = True
                    if ctx and ctx.get("frame") is not None and sp.last_seen == t:
                        sp.squat_frame, sp.squat_t = ctx["frame"].copy(), t
                elif t - sp.last_seen > cfg["max_gap_seconds"]:
                    sp.done = True  # dog just walked through
                continue
            # Suspected fouling: watch the spot for someone bending to clean up.
            for xyxy, tid in people:
                bx, by = _bottom_center(xyxy)
                if np.hypot(bx - sp.center[0], by - sp.center[1]) <= cfg["near_person_ratio"] * max(sp.diag, xyxy[3] - xyxy[1]):
                    sp.people_last_near = t
                    standing = self.person_height.get(tid, 0.0)
                    if tid >= 0 and standing > 0:
                        sp.min_height_ratio = min(sp.min_height_ratio, float(xyxy[3] - xyxy[1]) / standing)
            dog_near = any(np.linalg.norm(np.array(_center(x)) - sp.center) <= 2 * sp.diag for x, _ in dogs)
            left_at = max(sp.last_seen, sp.people_last_near)
            if dog_near or t - left_at < cfg["leave_seconds"]:
                continue
            sp.done = True
            cleaned = sp.min_height_ratio < cfg["bend_height_ratio"]
            if cleaned and not cfg["report_cleaned"]:
                continue
            dwell = sp.last_seen - sp.first_seen
            score = round(min(1.0, 0.4 * sp.conf_sum / sp.hits + 0.3 * min(1.0, dwell / (2 * cfg["min_seconds"])) + (0.0 if cleaned else 0.3)), 3)
            out.append(Candidate(
                rule=self.name, use_case="dog_fouling" if cleaned else "dog_fouling_not_cleaned",
                t_start=sp.first_seen, t_trigger=t, zone=sp.zone, score=score, bbox=sp.bbox,
                entities=[{"spot": [round(float(v)) for v in sp.center], "class": "dog"}],
                explanation_codes=["dog_stayed_on_spot"] + (["person_bent_down"] if cleaned else ["left_without_bending"]),
                extra={"dwell_seconds": round(dwell, 1), "lowest_person_height_ratio": round(sp.min_height_ratio, 2),
                       "left_at": round(left_at, 1)},
                moment_frame=sp.squat_frame, moment_t=sp.squat_t,
            ))
        self.spots = [sp for sp in self.spots if not sp.done or t - sp.last_seen < 30]
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

    def update(self, t: float, dets: sv.Detections, ctx: dict | None = None) -> list[Candidate]:
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

    def update(self, t: float, dets: sv.Detections, ctx: dict | None = None) -> list[Candidate]:
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


class _BinZone:
    def __init__(self, name: str, bin_box: list[float] | None, rect: list[int]):
        self.name, self.bin_box, self.rect = name, bin_box, rect
        self.baseline: np.ndarray | None = None
        self.baseline_mask: np.ndarray | None = None  # pixels hidden at baseline time
        self.visit_start: float | None = None
        self.last_occupied = -1e9
        self.changed_frames = 0


class BinDumpingRule:
    name = "bin_dumping"

    def __init__(self, cam: CameraConfig):
        self.cfg = cam.rule(self.name)
        self.manual = [z for z in cam.zones if "bin" in z.tags]
        self.zones: list[_BinZone] = []

    def _setup(self, frame: np.ndarray, bins: list) -> None:
        h, w = frame.shape[:2]
        self.zones = []
        for i, (b, _score) in enumerate(bins):
            bw, bh = b[2] - b[0], b[3] - b[1]
            side = self.cfg["side_margin"] * bw
            rect = [int(max(0, b[0] - side)), int(max(0, b[1] - 0.2 * bh)),
                    int(min(w, b[2] + side)), int(min(h, b[3] + self.cfg["floor_margin"] * bh))]
            self.zones.append(_BinZone(f"bin-{i + 1}", b, rect))
        for z in self.manual:
            x1, y1 = z.polygon_px.min(axis=0)
            x2, y2 = z.polygon_px.max(axis=0)
            self.zones.append(_BinZone(z.name, None, [int(x1), int(y1), int(x2), int(y2)]))

    @staticmethod
    def _people(dets: sv.Detections) -> np.ndarray:
        # Only people count as a visit. Parked cars are part of the scene; a
        # car that drives off after the event is handled by resetting the
        # baseline as soon as the decision is made.
        return dets.xyxy[dets.class_id == PERSON] if len(dets) else np.empty((0, 4))

    def _mask(self, z: _BinZone, boxes: np.ndarray, pad: float = 0.15) -> np.ndarray:
        x1, y1, x2, y2 = z.rect
        m = np.zeros((y2 - y1, x2 - x1), np.uint8)
        for b in boxes:
            bw, bh = b[2] - b[0], b[3] - b[1]
            a1, b1 = int(b[0] - pad * bw) - x1, int(b[1] - pad * bh) - y1
            a2, b2 = int(b[2] + pad * bw) - x1, int(b[3] + pad * bh) - y1
            m[max(0, b1):max(0, b2), max(0, a1):max(0, a2)] = 1
        return m

    def update(self, t: float, dets: sv.Detections, ctx: dict | None = None) -> list[Candidate]:
        if ctx is None:
            return []
        frame = ctx["frame"]
        if ctx.get("new_scene"):
            self._setup(frame, ctx.get("bins", []))
        cfg = self.cfg
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
        people = self._people(dets)
        out = []
        for z in self.zones:
            x1, y1, x2, y2 = z.rect
            crop = gray[y1:y2, x1:x2]
            occ_mask = self._mask(z, people)
            occupied = occ_mask.mean() > 0.02
            if z.baseline is None:
                # First look at this zone. Pixels hidden by a person stay
                # masked until they are seen empty.
                z.baseline, z.baseline_mask = crop.copy(), occ_mask
            elif z.visit_start is None:
                # Nobody has visited yet: keep learning the empty look.
                free = occ_mask == 0
                z.baseline[free] = 0.9 * z.baseline[free] + 0.1 * crop[free]
                z.baseline_mask = z.baseline_mask & occ_mask
            if occupied:
                z.last_occupied = t
                if z.visit_start is None:
                    z.visit_start = t
                z.changed_frames = 0
                continue
            if z.visit_start is None or t - z.last_occupied < cfg["settle_seconds"]:
                continue
            diff = np.abs((crop - np.median(crop)) - (z.baseline - np.median(z.baseline)))
            changed = (diff > cfg["diff_threshold"]).astype(np.uint8)
            changed[z.baseline_mask > 0] = 0
            if z.bin_box is not None:
                # Ignore the lid and upper body (using the bin properly is fine,
                # and lids move), but keep its base: bags get dumped against it.
                b = z.bin_box
                lid_bottom = int(b[1] + 0.55 * (b[3] - b[1]))
                changed[max(0, int(b[1]) - y1):max(0, lid_bottom - y1), max(0, int(b[0]) - x1):max(0, int(b[2]) - x1)] = 0
            changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, _lab, stats, _ = cv2.connectedComponentsWithStats(changed)
            best = max(range(1, n), key=lambda k: stats[k, 4], default=None)
            area = int(stats[best, 4]) if best is not None else 0
            z.last_area = area
            if area >= max(cfg["min_area_px"], cfg["min_area_fraction"] * crop.size):
                z.changed_frames += 1
            else:
                z.changed_frames = 0
            if z.changed_frames >= cfg["confirm_frames"]:
                bx, by, bw, bh = stats[best, :4]
                box = [float(x1 + bx), float(y1 + by), float(x1 + bx + bw), float(y1 + by + bh)]
                out.append(Candidate(
                    rule=self.name, use_case="littering", t_start=z.visit_start, t_trigger=t, zone=z.name,
                    score=round(min(1.0, 0.5 + area / crop.size), 3), bbox=box,
                    entities=[{"object_id": f"{z.name}-change", "class": "unknown_item"}],
                    explanation_codes=["person_visited_bin", "person_left", "new_item_beside_bin"],
                    extra={"changed_px": area, "zone_rect": z.rect},
                ))
            if z.changed_frames >= cfg["confirm_frames"] or t - z.last_occupied > cfg["settle_seconds"] + 3:
                z.visit_start, z.changed_frames = None, 0
                z.baseline, z.baseline_mask = crop.copy(), occ_mask
        return out


ALL_RULES = [DogFoulingRule, ObjectLeftBehindRule, PersonInZoneRule, BinDumpingRule]


def build_rules(cam: CameraConfig) -> list:
    return [r(cam) for r in ALL_RULES if cam.rule(r.name).get("enabled")]


def classes_needed(cam: CameraConfig) -> list[str]:
    names = {"person", "dog"}
    if cam.rule("object_left_behind").get("enabled"):
        names |= set(cam.rule("object_left_behind")["classes"])
    if cam.rule("bin_dumping").get("enabled"):
        names |= {"car", "truck", "bus", "motorcycle", "bicycle"}
    return sorted(names)
