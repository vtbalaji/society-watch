"""Per-run summary in Markdown and a self-contained HTML review page."""

from __future__ import annotations

import html
from pathlib import Path


def _fmt_t(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:04.1f}"


def write_markdown(results: list[dict], out_root: Path) -> Path:
    lines = ["# Incident candidates", "", "All events are **candidates for human review**, not confirmed incidents.", ""]
    lines += ["| Camera | Video length | Processed at | Candidates | By type |", "|---|---|---|---|---|"]
    for r in results:
        s = r["summary"]
        by = ", ".join(f"{k}: {v}" for k, v in s["events_by_use_case"].items()) or "none"
        lines.append(f"| {s['camera_id']} | {_fmt_t(s['video']['duration_seconds'])} | {s['processing']['realtime_factor']}x realtime | {s['event_count']} | {by} |")
    for r in results:
        s = r["summary"]
        a = s["activity"]
        lines += ["", f"## {s['camera_id']}", "",
                  f"Source `{s['source']}`, {s['video']['resolution'][0]}x{s['video']['resolution'][1]}, "
                  f"sampled at {s['processing']['sample_fps']} fps with {s['processing']['detector']}.",
                  f"People visible {a['seconds_with_person']}s (max {a['max_people_in_frame']} at once), "
                  f"dogs visible {a['seconds_with_dog']}s (max {a['max_dogs_in_frame']}).", ""]
        if not r["events"]:
            lines.append("No candidates.")
            continue
        lines += ["| Time | Type | Zone | Score | Why | Triage | Clip |", "|---|---|---|---|---|---|---|"]
        for e in r["events"]:
            tri = e.get("triage", {})
            lines.append(
                f"| {_fmt_t(e['video_offset_trigger'])} | {e['use_case']} | {e['zone']} | {e['score']:.2f} | "
                f"{', '.join(e['explanation_codes'])} | {tri.get('verdict', '-')} | {e['evidence']['clip'] or '-'} |")
    path = out_root / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_html(results: list[dict], out_root: Path) -> Path:
    cards = []
    for r in results:
        s = r["summary"]
        cards.append(f"<h2>{html.escape(s['camera_id'])} <small>{s['event_count']} candidates, "
                     f"{_fmt_t(s['video']['duration_seconds'])} of video</small></h2>")
        if not r["events"]:
            cards.append("<p class=muted>No candidates.</p>")
        cards.append("<div class=grid>")
        for e in r["events"]:
            tri = e.get("triage")
            tri_html = f"<p><b>VLM:</b> {html.escape(tri['verdict'])} &middot; {html.escape(tri['answer'][:200])}</p>" if tri else ""
            clip = e["evidence"]["clip"]
            media = (f"<video controls preload=none poster='{html.escape(e['evidence']['keyframes'][0])}' src='{html.escape(clip)}'></video>"
                     if clip else f"<img src='{html.escape(e['evidence']['keyframes'][0])}'>")
            cards.append(
                f"<div class=card>{media}<p><b>{html.escape(e['use_case'])}</b> at {_fmt_t(e['video_offset_trigger'])} "
                f"in <i>{html.escape(e['zone'])}</i>, score {e['score']:.2f}</p>"
                f"<p class=muted>{html.escape(', '.join(e['explanation_codes']))}</p>{tri_html}</div>")
        cards.append("</div>")
    page = f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Incident candidates</title>
<style>
:root{{--bg:#fafafa;--fg:#1a1a1a;--muted:#666;--card:#fff;--line:#ddd}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151515;--fg:#eee;--muted:#999;--card:#1f1f1f;--line:#333}}}}
body{{font:15px/1.45 -apple-system,system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0 auto;max-width:1200px;padding:16px}}
small,.muted{{color:var(--muted);font-weight:normal}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px}}
.card video,.card img{{width:100%;border-radius:4px;background:#000}}
.card p{{margin:6px 0}}
</style>
<h1>Incident candidates</h1>
<p class=muted>Every item is a candidate for human review, not a confirmed incident.</p>
{''.join(cards)}
"""
    path = out_root / "report.html"
    path.write_text(page)
    return path
