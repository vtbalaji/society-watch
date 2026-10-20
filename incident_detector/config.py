"""Camera and rule configuration.

A config file lists cameras. Each camera has a video source, zones drawn as
polygons in normalised (0-1) coordinates so they survive resolution changes,
and per-rule settings that override the defaults below.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

DEFAULTS: dict = {
    "sample_fps": 5.0,
    "model": "medium",  # nano | small | medium | base | large
    "clip_pre_seconds": 8.0,
    "clip_post_seconds": 12.0,
    "rules": {
        "dog_dwell": {
            "enabled": True,
            "zones": ["*"],  # zone names, or "*" for every zone
            "min_seconds": 5.0,
            # Max movement of the dog's centre over the dwell window, as a
            # fraction of its own box diagonal. Sniffing/squatting stays low.
            "max_motion_ratio": 0.6,
            "min_confidence": 0.4,
            "cooldown_seconds": 30.0,
        },
        "object_left_behind": {
            "enabled": True,
            "zones": ["*"],
            "classes": ["bottle", "cup", "handbag", "backpack", "suitcase", "umbrella", "sports ball"],
            "litter_classes": ["bottle", "cup", "sports ball"],
            "min_confidence": 0.3,
            "static_seconds": 8.0,  # object must stay put this long
            "unattended_seconds": 8.0,  # with no person close for this long
            "near_person_ratio": 0.8,  # "close" = within this x person height
            "require_owner": True,  # a person must have been near it when it appeared
            "warmup_seconds": 3.0,  # objects already there at start are background
            "cooldown_seconds": 120.0,
        },
        "person_in_restricted_zone": {
            "enabled": False,
            "zones": [],
            "min_seconds": 10.0,
            "min_confidence": 0.5,
            "cooldown_seconds": 60.0,
        },
    },
}


@dataclass
class Zone:
    name: str
    polygon_norm: list[list[float]]
    tags: list[str] = field(default_factory=list)
    polygon_px: np.ndarray | None = None

    def scale(self, width: int, height: int) -> None:
        pts = np.array(self.polygon_norm, dtype=float) * [width, height]
        self.polygon_px = pts.astype(np.int64)


@dataclass
class CameraConfig:
    camera_id: str
    source: str
    site_id: str = "site-1"
    start_time: str | None = None  # ISO time of first frame, if known
    zones: list[Zone] = field(default_factory=list)
    settings: dict = field(default_factory=dict)

    def rule(self, name: str) -> dict:
        return self.settings["rules"][name]

    def zones_for(self, rule_name: str) -> list[Zone]:
        wanted = self.rule(rule_name).get("zones", ["*"])
        if "*" in wanted:
            return self.zones
        return [z for z in self.zones if z.name in wanted or set(z.tags) & set(wanted)]


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


WHOLE_FRAME = [[0, 0], [1, 0], [1, 1], [0, 1]]


def _camera(cam_id: str, raw: dict, defaults: dict, base_dir: Path) -> CameraConfig:
    settings = _deep_merge(defaults, {k: v for k, v in raw.items() if k in ("sample_fps", "model", "clip_pre_seconds", "clip_post_seconds", "rules")})
    zones = [Zone(z["name"], z["polygon"], z.get("tags", [])) for z in raw.get("zones", [])]
    if not zones:
        zones = [Zone("whole_frame", WHOLE_FRAME, ["common_area"])]
    source = raw["source"]
    if "://" not in source and not Path(source).is_absolute():
        source = str((base_dir / source).resolve())
    return CameraConfig(cam_id, source, raw.get("site_id", "site-1"), raw.get("start_time"), zones, settings)


def load_config(path: str | Path) -> list[CameraConfig]:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    defaults = _deep_merge(DEFAULTS, data.get("defaults", {}))
    base_dir = path.parent.parent if path.parent.name == "configs" else path.parent
    return [_camera(cid, raw, defaults, base_dir) for cid, raw in data["cameras"].items()]


def single_file_config(source: str, camera_id: str | None = None) -> CameraConfig:
    cam_id = camera_id or Path(source).stem
    return _camera(cam_id, {"source": str(Path(source).resolve())}, DEFAULTS, Path.cwd())
