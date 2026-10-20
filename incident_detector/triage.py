"""Optional second opinion from a small local VLM (mlx-vlm) on flagged events.

Only flagged keyframes are sent, never the full video. The VLM's answer is a
reviewer aid stored next to the event; it does not change the event status.
Install with:  uv pip install --python .venv mlx-vlm
"""

from __future__ import annotations

from pathlib import Path

import cv2

DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"

QUESTIONS = {
    "dog_fouling": "Look at the dog inside the red box. Is the dog defecating or squatting to poop? "
                   "Answer YES, NO or UNSURE on the first line, then one short sentence of reasoning.",
    "littering": "Look at the object inside the red box. Does it look like litter or rubbish left on the ground? "
                 "Answer YES, NO or UNSURE on the first line, then one short sentence of reasoning.",
    "unattended_item": "Look at the object inside the red box. Does it look like a bag or item left behind with nobody attending it? "
                       "Answer YES, NO or UNSURE on the first line, then one short sentence of reasoning.",
    "restricted_zone": "Is there a person standing or walking in this area? "
                       "Answer YES, NO or UNSURE on the first line, then one short sentence of reasoning.",
}


def context_crop(image_path: Path, bbox: list[float], out_path: Path, max_side: int = 768) -> Path:
    """Crop around the flagged box with generous context; small VLMs do far
    better on a focused crop than on a full 1080p/4K frame."""
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh) * 2.5
    side = max(side, min(w, h) * 0.4)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left, top = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    right, bottom = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    crop = img[top:bottom, left:right]
    scale = max_side / max(crop.shape[:2])
    if scale < 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), crop)
    return out_path


class Triage:
    def __init__(self, model_id: str = DEFAULT_MODEL):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        self.model_id = model_id
        self.model, self.processor = load(model_id)
        self.config = load_config(model_id)

    def ask(self, image_path: Path, use_case: str) -> dict:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        question = QUESTIONS.get(use_case, QUESTIONS["littering"])
        prompt = apply_chat_template(self.processor, self.config, question, num_images=1)
        result = generate(self.model, self.processor, prompt, [str(image_path)], max_tokens=60, verbose=False)
        text = (getattr(result, "text", result) or "").strip()
        first = text.split()[0].strip(".,:").upper() if text else ""
        verdict = first if first in ("YES", "NO", "UNSURE") else "UNSURE"
        return {"model": self.model_id, "verdict": verdict, "answer": text}


def triage_events(results: list[dict], out_root: Path, model_id: str = DEFAULT_MODEL) -> None:
    events = [e for r in results for e in r["events"]]
    if not events:
        return
    t = Triage(model_id)
    for e in events:
        kf = out_root / e["evidence"]["keyframes"][0]
        crop = context_crop(kf, e["bbox"], kf.with_name(kf.stem + "_crop.jpg"))
        e["triage"] = t.ask(crop, e["use_case"])
        e["evidence"]["keyframes"].append(str(crop.relative_to(out_root)))
        print(f"  triage {e['camera_id']} {e['use_case']} @{e['video_offset_trigger']}s -> {e['triage']['verdict']}")
