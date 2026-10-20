"""Batch pipeline: decode -> sample -> detect -> track -> rules -> evidence."""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from . import RULES_VERSION, __version__
from .config import CameraConfig
from .detector import Detector, class_name
from .rules import DOG, PERSON, Candidate, build_rules, classes_needed

MERGE_SECONDS = 15.0


def _annotate(frame: np.ndarray, dets: sv.Detections, cam: CameraConfig, highlight: list[float] | None = None) -> np.ndarray:
    img = frame.copy()
    for z in cam.zones:
        cv2.polylines(img, [z.polygon_px.reshape(-1, 1, 2).astype(np.int32)], True, (0, 200, 255), 2)
        x, y = z.polygon_px[0]
        cv2.putText(img, z.name, (int(x) + 6, int(y) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    if len(dets):
        labels = [
            f"{class_name(c)} {p:.2f}" + (f" #{t}" if t is not None and t >= 0 else "")
            for c, p, t in zip(dets.class_id, dets.confidence, dets.tracker_id if dets.tracker_id is not None else [None] * len(dets))
        ]
        img = sv.BoxAnnotator(thickness=2).annotate(img, dets)
        img = sv.LabelAnnotator(text_scale=0.5).annotate(img, dets, labels=labels)
    if highlight is not None:
        x1, y1, x2, y2 = map(int, highlight)
        cv2.rectangle(img, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), (0, 0, 255), 4)
    return img


def _merge(cands: list[tuple[Candidate, np.ndarray]]) -> list[tuple[Candidate, np.ndarray]]:
    merged: list[tuple[Candidate, np.ndarray]] = []
    for c, img in sorted(cands, key=lambda x: x[0].t_trigger):
        prev = next((m for m in reversed(merged) if m[0].rule == c.rule and m[0].zone == c.zone), None)
        if prev and c.t_start - prev[0].t_trigger <= MERGE_SECONDS and c.rule != "object_left_behind":
            p = prev[0]
            p.t_trigger = max(p.t_trigger, c.t_trigger)
            p.extra["merged_triggers"] = p.extra.get("merged_triggers", 1) + 1
            p.entities += [e for e in c.entities if e not in p.entities]
            if c.score > p.score:
                p.score, p.bbox = c.score, c.bbox
                merged[merged.index(prev)] = (p, img)
            continue
        merged.append((c, img))
    return merged


def _cut_clip(src: str, start: float, end: float, dst: Path) -> bool:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{max(0.0, start):.2f}", "-i", src, "-t", f"{end - max(0.0, start):.2f}",
        "-an", "-c:v", "h264_videotoolbox", "-b:v", "4M", "-movflags", "+faststart", str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:  # fall back to software encode
        cmd[cmd.index("h264_videotoolbox")] = "libx264"
        r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0


def _iso(cam: CameraConfig, offset: float) -> str | None:
    if not cam.start_time:
        return None
    base = datetime.fromisoformat(cam.start_time.replace("Z", "+00:00"))
    return (base + timedelta(seconds=offset)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def process_camera(cam: CameraConfig, out_root: Path, detector_cache: dict, annotate_video: bool = False,
                   max_seconds: float | None = None, progress: bool = True) -> dict:
    cap = cv2.VideoCapture(cam.source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source for camera {cam.camera_id}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for z in cam.zones:
        z.scale(width, height)

    s = cam.settings
    stride = max(1, round(fps / s["sample_fps"]))
    eff_fps = fps / stride
    key = (s["model"], tuple(classes_needed(cam)))
    if key not in detector_cache:
        detector_cache[key] = Detector(s["model"], list(key[1]))
    det = detector_cache[key]
    tracker = ByteTrackTracker(frame_rate=eff_fps, lost_track_buffer=int(eff_fps * 3),
                               track_activation_threshold=0.4, high_conf_det_threshold=0.5)
    rules = build_rules(cam)

    out_dir = out_root / cam.camera_id
    (out_dir / "clips").mkdir(parents=True, exist_ok=True)
    (out_dir / "keyframes").mkdir(exist_ok=True)
    writer = None
    if annotate_video:
        writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"avc1"), eff_fps, (width, height))

    cands: list[tuple[Candidate, np.ndarray]] = []
    stats = Counter()
    max_people = max_dogs = 0
    t_start = time.time()
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = idx / fps
            if max_seconds and t > max_seconds:
                break
            dets = tracker.update(det(frame), timestamp=t)
            if dets.tracker_id is None:
                dets.tracker_id = np.full(len(dets), -1)
            stats["sampled_frames"] += 1
            n_p, n_d = int(np.sum(dets.class_id == PERSON)), int(np.sum(dets.class_id == DOG))
            stats["frames_with_person"] += n_p > 0
            stats["frames_with_dog"] += n_d > 0
            max_people, max_dogs = max(max_people, n_p), max(max_dogs, n_d)
            for c in dets.class_id:
                stats[f"det_{class_name(c)}"] += 1
            for rule in rules:
                for c in rule.update(t, dets):
                    cands.append((c, _annotate(frame, dets, cam, c.bbox)))
            if writer is not None:
                writer.write(_annotate(frame, dets, cam))
            if progress and stats["sampled_frames"] % 50 == 0:
                rate = stats["sampled_frames"] / (time.time() - t_start)
                print(f"  [{cam.camera_id}] {t:7.1f}s / {total / fps:.1f}s  {rate:.1f} frames/s  candidates={len(cands)}", flush=True)
        idx += 1
    cap.release()
    if writer is not None:
        writer.release()
    elapsed = time.time() - t_start
    duration = idx / fps

    events = []
    for c, img in _merge(cands):
        eid = str(uuid.uuid4())
        clip_start, clip_end = max(0.0, c.t_start - s["clip_pre_seconds"]), min(duration, c.t_trigger + s["clip_post_seconds"])
        clip = out_dir / "clips" / f"{c.use_case}_{c.t_trigger:07.1f}s_{eid[:8]}.mp4"
        kf = out_dir / "keyframes" / f"{clip.stem}.jpg"
        cv2.imwrite(str(kf), img)
        clip_ok = _cut_clip(cam.source, clip_start, clip_end, clip)
        events.append({
            "event_id": eid,
            "site_id": cam.site_id,
            "camera_id": cam.camera_id,
            "use_case": c.use_case,
            "rule": c.rule,
            "zone": c.zone,
            "video_offset_start": round(c.t_start, 2),
            "video_offset_trigger": round(c.t_trigger, 2),
            "event_time_start": _iso(cam, c.t_start),
            "event_time_end": _iso(cam, c.t_trigger),
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "needs_review",
            "score": c.score,
            "bbox": [round(v, 1) for v in c.bbox],
            "evidence": {
                "clip": str(clip.relative_to(out_root)) if clip_ok else None,
                "clip_window": [round(clip_start, 2), round(clip_end, 2)],
                "keyframes": [str(kf.relative_to(out_root))],
            },
            "entities": c.entities,
            "explanation_codes": c.explanation_codes,
            "details": c.extra,
            "versions": {"detector": det.name, "rules": RULES_VERSION, "pipeline": __version__},
            "source": {"mode": "batch", "uri": Path(cam.source).name},
            "schema_version": 1,
        })

    with open(out_dir / "events.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    summary = {
        "camera_id": cam.camera_id,
        "site_id": cam.site_id,
        "source": Path(cam.source).name,
        "video": {"duration_seconds": round(duration, 1), "fps": round(fps, 2), "resolution": [width, height]},
        "processing": {
            "sample_fps": round(eff_fps, 2),
            "sampled_frames": stats["sampled_frames"],
            "wall_seconds": round(elapsed, 1),
            "frames_per_second": round(stats["sampled_frames"] / elapsed, 2) if elapsed else None,
            "realtime_factor": round(duration / elapsed, 2) if elapsed else None,
            "detector": det.name,
            "rules": [r.name for r in rules],
        },
        "activity": {
            "seconds_with_person": round(stats["frames_with_person"] / eff_fps, 1),
            "seconds_with_dog": round(stats["frames_with_dog"] / eff_fps, 1),
            "max_people_in_frame": max_people,
            "max_dogs_in_frame": max_dogs,
            "detections_by_class": {k[4:]: v for k, v in sorted(stats.items()) if k.startswith("det_")},
        },
        "events_by_use_case": dict(Counter(e["use_case"] for e in events)),
        "event_count": len(events),
        "events": [{k: e[k] for k in ("event_id", "use_case", "zone", "video_offset_trigger", "score", "evidence")} for e in events],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return {"summary": summary, "events": events}
