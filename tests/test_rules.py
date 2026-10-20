"""Rule tests on synthetic detections (no model or video needed)."""

import numpy as np
import supervision as sv

from incident_detector.config import single_file_config
from incident_detector.detector import NAME_TO_ID
from incident_detector.rules import DogFoulingRule, ObjectLeftBehindRule

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


def _dog_scene(owner_bends: bool):
    """Dog squats at one spot for 5 s with its owner beside it, then both walk off."""
    frames = []
    for i in range(FPS * 20):
        t = i / FPS
        items = []
        if t < 2:  # walking in
            x = 300 + t * 100
            items += [("dog", [x, 400, x + 100, 480], 1), ("person", [x + 120, 250, x + 180, 480], 2)]
        elif t < 7:  # squatting; owner waits (and maybe bends)
            items.append(("dog", [500 + (i % 2) * 3, 400, 600, 480], 1))
            top = 380 if (owner_bends and 5 <= t < 6) else 250
            items.append(("person", [620, top, 680, 480], 2))
        elif t < 10:  # walking away
            x = 500 + (t - 7) * 250
            items += [("dog", [x, 400, x + 100, 480], 1), ("person", [x + 120, 250, x + 180, 480], 2)]
        frames.append(dets(*items))
    return run(DogFoulingRule(cam()), frames)


def test_dog_fouling_not_cleaned_fires():
    events = _dog_scene(owner_bends=False)
    assert len(events) == 1
    assert events[0].use_case == "dog_fouling_not_cleaned"
    assert "left_without_bending" in events[0].explanation_codes


def test_owner_bending_down_counts_as_cleaned():
    assert _dog_scene(owner_bends=True) == []


def test_walking_dog_does_not_fire():
    rule = DogFoulingRule(cam())
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


def _bin_frames(drop: bool):
    """Grey floor, a bin at (600,400)-(640,480); a person visits 2-6 s."""
    from incident_detector.rules import BinDumpingRule

    rule = BinDumpingRule(cam())
    bins = [([600, 400, 640, 480], 0.9)]
    events = []
    for i in range(FPS * 20):
        t = i / FPS
        frame = np.full((H, W, 3), 120, np.uint8)
        frame[400:480, 600:640] = (0, 200, 220)  # the bin
        items = []
        if 2 <= t < 6:
            items.append(("person", [660, 300, 720, 500], 1))
            frame[300:500, 660:720] = 60
        if drop and t >= 5:
            frame[455:480, 645:670] = 240  # white bag at the bin's base
        events += rule.update(t, dets(*items), {"frame": frame, "new_scene": i == 0, "bins": bins})
    return events


def test_bag_left_beside_bin_fires():
    events = _bin_frames(drop=True)
    assert len(events) == 1
    assert events[0].rule == "bin_dumping"
    assert 6 <= events[0].t_trigger <= 10


def test_bin_visit_without_drop_is_quiet():
    assert _bin_frames(drop=False) == []
