# society-watch

Apartment incident detector (phase 1).
Batch pipeline that scans recorded camera footage for **candidate** rule-breaking
in common areas and writes short evidence clips plus a per-camera summary.
Everything it flags is a candidate for a person to review, not a confirmed incident.

Runs locally on an Apple Silicon Mac using only open-source parts:

| Stage | Tool | Licence |
|---|---|---|
| Detect people, dogs, bags, bottles, cups | RF-DETR Medium (COCO), traced fp16 on the Apple GPU | Apache-2.0 |
| Track across frames | ByteTrack from Roboflow `trackers` | Apache-2.0 |
| Zones, annotation | Roboflow Supervision | MIT |
| Optional second opinion | Qwen2.5-VL-3B (4-bit) via `mlx-vlm` | Apache-2.0 / MIT |
| Clip cutting | ffmpeg with VideoToolbox | LGPL/GPL build |

## Rules

| Rule | Fires when | Use case |
|---|---|---|
| `dog_dwell` | A dog stays nearly still inside a watched zone (lawn) for 5 s | `dog_fouling` |
| `object_left_behind` | A bottle/cup/bag appears next to a person, stays put 8 s, and nobody is near it for 8 s | `littering` or `unattended_item` |
| `person_in_restricted_zone` | A person stays in a restricted zone for N s (off by default) | `restricted_zone` |

Objects already in view during the first 3 s are treated as background, and a
carried bag never fires because it moves with its owner. Thresholds live in
`incident_detector/config.py` and can be overridden per camera.

## Setup

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev]"          # add ,triage for the VLM step
scripts/fetch_samples.sh                           # public test clips (optional)
```

The first run downloads RF-DETR weights (~390 MB) to `~/.roboflow/models`;
`--triage` downloads the VLM (~3 GB) to the Hugging Face cache.

## Run

```sh
# every camera in the config
.venv/bin/python -m incident_detector run configs/cameras.yaml

# one file, whole frame as the zone
.venv/bin/python -m incident_detector file ~/footage/lobby-2026-10-19.mp4 --camera-id lobby

# options: --camera ID  --annotate  --triage  --model nano|small|medium|large  --max-seconds 60
```

Each run writes `output/run-<timestamp>/`:

- `summary.md` and `report.html` (keyframe grid with playable clips)
- `<camera>/events.jsonl` (one event per line, schema from the platform spec)
- `<camera>/summary.json` (video length, speed, activity, events by type)
- `<camera>/clips/*.mp4` (8 s before to 12 s after each trigger) and `keyframes/*.jpg`
- `<camera>/annotated.mp4` with `--annotate`

## Adding a real camera

Add an entry to `configs/cameras.yaml`: the footage path, `start_time` (so events
get wall-clock times), and zones drawn as polygons in 0 to 1 coordinates over the
lawn, lobby floor, corridor and so on. Point each rule at the zones it should watch.

## Results on public test clips (M5, 5 fps sampling)

| Clip | Speed | Candidates | VLM triage |
|---|---|---|---|
| Dogs resting in grass (CC0) | 7.6x realtime | 1 dog_fouling | NO (dog standing) |
| Dog park opening, 14 people, 5 dogs (public domain) | 6.9x realtime | 1 dog_fouling | NO (dog standing) |
| People walking (Supervision sample) | 6.5x realtime | 1 restricted_zone (demo) | YES |

These clips contain no real fouling or littering, so they only show the
plumbing and that triage rejects the obvious false alarms. The left-behind
logic is covered by synthetic tests in `tests/test_rules.py`. Real footage
from the complex's cameras is needed to tune thresholds and measure precision.

## Tests

```sh
.venv/bin/python -m pytest -q
```
