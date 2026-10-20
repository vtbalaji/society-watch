"""CLI.

  python -m incident_detector run configs/cameras.yaml [--camera ID] [--triage]
  python -m incident_detector file path/to/video.mp4 [--camera-id ID]
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")  # torch tracing / transformers noise

from .config import load_config, single_file_config
from .pipeline import process_camera
from .report import write_html, write_markdown


def main() -> None:
    ap = argparse.ArgumentParser(prog="incident_detector")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="process every camera in a config file")
    run.add_argument("config")
    run.add_argument("--camera", action="append", help="only these camera ids")
    one = sub.add_parser("file", help="process one video with default rules over the whole frame")
    one.add_argument("video")
    one.add_argument("--camera-id")
    for p in (run, one):
        p.add_argument("--out", default="output")
        p.add_argument("--model", help="override detector size: nano|small|medium|base|large")
        p.add_argument("--max-seconds", type=float, help="stop after this much video (for quick tests)")
        p.add_argument("--stabilize", action="store_true", help="align shaky footage (e.g. a phone filming the NVR screen)")
        p.add_argument("--annotate", action="store_true", help="also write an annotated video per camera")
        p.add_argument("--triage", action="store_true", help="ask a local VLM about each flagged keyframe")
        p.add_argument("--triage-model", default=None)
    args = ap.parse_args()

    cams = load_config(args.config) if args.cmd == "run" else [single_file_config(args.video, args.camera_id)]
    if args.cmd == "run" and args.camera:
        cams = [c for c in cams if c.camera_id in args.camera]
    for c in cams:
        if args.model:
            c.settings["model"] = args.model
        if args.stabilize:
            c.settings["stabilize"] = True

    out_root = Path(args.out) / datetime.now().strftime("run-%Y%m%d-%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    results = []
    for cam in cams:
        print(f"Processing {cam.camera_id}: {cam.source}")
        results.append(process_camera(cam, out_root, cache, annotate_video=args.annotate, max_seconds=args.max_seconds))
        s = results[-1]["summary"]
        print(f"  done: {s['event_count']} candidates {s['events_by_use_case']}, {s['processing']['realtime_factor']}x realtime")

    if args.triage:
        from .triage import DEFAULT_MODEL, triage_events

        triage_events(results, out_root, args.triage_model or DEFAULT_MODEL)
        for r in results:
            cam_dir = out_root / r["summary"]["camera_id"]
            (cam_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in r["events"]))

    md, page = write_markdown(results, out_root), write_html(results, out_root)
    print(f"\nSummary: {md}\nReview page: {page}")


if __name__ == "__main__":
    main()
