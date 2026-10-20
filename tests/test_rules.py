"""Rule tests on synthetic detections (no model or video needed)."""

import numpy as np
import supervision as sv

from incident_detector.config import single_file_config
from incident_detector.detector import NAME_TO_ID
from incident_detector.rules import DogDwellRule, ObjectLeftBehindRule

W, H, FPS = 1280, 720, 5


def cam():
    c = single_file_config("dummy.mp4", "test")
    for z in c.zones:
        z.scale(W, H)
    return c


def dets(*items):
    """items: (class_name, xyxy, track_id)"""
    if not items:
        return sv.Detections(xyxy=np.empty((0, 4)), class_id=np.empty(0, int), confidence=np.empty(0), tracker_id=np.empty(0, int))
    return sv.Detections(
        xyxy=np.array([i[1] for i in items], float),
        class_id=np.array([NAME_TO_ID[i[0]] for i in items]),
        confidence=np.full(len(items), 0.9),
        tracker_id=np.array([i[2] for i in items]),
    )


def run(rule, frames):
    out = []
    for i, d in enumerate(frames):
        out += rule.update(i / FPS, d)
    return out


def test_still_dog_fires_once():
    rule = DogDwellRule(cam())
    box = [500, 400, 600, 480]
    events = run(rule, [dets(("dog", [box[0] + (i % 2) * 3, box[1], box[2] + (i % 2) * 3, box[3]], 1)) for i in range(FPS * 12)])
    assert len(events) == 1
    assert events[0].use_case == "dog_fouling"
    assert 4.5 <= events[0].t_trigger <= 5.5


def test_walking_dog_does_not_fire():
    rule = DogDwellRule(cam())
    events = run(rule, [dets(("dog", [100 + i * 15, 400, 200 + i * 15, 480], 1)) for i in range(FPS * 10)])
    assert events == []


def test_bottle_left_behind_after_person_leaves():
    rule = ObjectLeftBehindRule(cam())
    bottle = [600, 600, 630, 650]
    frames = []
    for i in range(FPS * 30):
        t = i / FPS
        items = []
        if t < 6:  # person walks up and stands next to the spot
            items.append(("person", [560, 300, 680, 660], 1))
        elif t < 9:  # person walks away
            x = 560 + (t - 6) * 200
            items.append(("person", [x, 300, x + 120, 660], 1))
        if t >= 4:  # bottle appears while the person is there and stays
            items.append(("bottle", bottle, 2))
        frames.append(dets(*items))
    events = run(rule, frames)
    assert len(events) == 1
    e = events[0]
    assert e.use_case == "littering"
    assert "person_was_near" in e.explanation_codes
    assert e.t_trigger >= 4 + rule.cfg["static_seconds"]


def test_object_present_from_start_is_background():
    rule = ObjectLeftBehindRule(cam())
    frames = [dets(("backpack", [600, 600, 660, 680], 1)) for _ in range(FPS * 30)]
    assert run(rule, frames) == []


def test_carried_bag_does_not_fire():
    rule = ObjectLeftBehindRule(cam())
    frames = []
    for i in range(FPS * 30):
        x = 100 + i * 8
        frames.append(dets(("person", [x, 300, x + 120, 660], 1), ("handbag", [x + 80, 450, x + 130, 520], 2)))
    assert run(rule, frames) == []
