r"""Render a meeting/session step timeline as a standalone HTML file.

Example:
    C:\Users\LX034\miniconda3\python.exe examples\visualize_session_timeline.py ^
        --meeting-id mtg_14bf5d2086

The horizontal axis is the meeting step, not wall-clock time. Within each step, a
turn's LLM/tool/session events are spread by their recorded sequence or message index.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def iso(value: datetime | None) -> str | None:
    return value.isoformat(timespec="milliseconds") if value else None


def shorten(value: Any, max_len: int = 220) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "..."


def session_key(path: str | None, agent: str | None, round_num: int | None) -> str | None:
    if path:
        return Path(path).name
    if agent and round_num is not None:
        return f"{agent}_r{round_num}.json"
    return None


def load_sessions(sessions_dir: Path) -> dict[str, dict[str, Any]]:
    sessions: dict[str, dict[str, Any]] = {}
    if not sessions_dir.exists():
        return sessions

    for path in sorted(sessions_dir.glob("*.json")):
        messages = json.loads(path.read_text(encoding="utf-8"))
        role_counts = Counter(m.get("role", "unknown") for m in messages)
        tool_calls = 0
        tool_results = 0
        tools: Counter[str] = Counter()
        content_chars = 0
        derived: list[dict[str, Any]] = []

        for index, message in enumerate(messages):
            role = message.get("role", "unknown")
            content = message.get("content") or ""
            content_chars += len(content)

            calls = message.get("tool_calls") or []
            if calls:
                for call in calls:
                    fn = (call.get("function") or {}).get("name", "tool")
                    tool_calls += 1
                    tools[fn] += 1
                    derived.append({
                        "message_index": index,
                        "type": "tool_request",
                        "label": fn,
                        "detail": shorten((call.get("function") or {}).get("arguments")),
                        "input": (call.get("function") or {}).get("arguments"),
                        "output": "",
                    })
            elif role == "tool":
                name = message.get("name", "tool")
                tool_results += 1
                tools[name] += 1
                derived.append({
                    "message_index": index,
                    "type": "tool_result",
                    "label": name,
                    "detail": shorten(content),
                    "input": "",
                    "output": content,
                })
            elif role == "assistant" and content.strip():
                derived.append({
                    "message_index": index,
                    "type": "assistant_text",
                    "label": "assistant",
                    "detail": shorten(content),
                    "input": "",
                    "output": content,
                })
            elif role == "user":
                derived.append({
                    "message_index": index,
                    "type": "user_message",
                    "label": "user",
                    "detail": shorten(content),
                    "input": content,
                    "output": "",
                })

        sessions[path.name] = {
            "path": str(path),
            "message_count": len(messages),
            "role_counts": dict(role_counts),
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "tools": dict(sorted(tools.items())),
            "content_chars": content_chars,
            "derived_events": derived,
            "messages": messages,
        }
    return sessions


def build_timeline(run: dict[str, Any], sessions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    lanes: list[str] = []
    lane_seen: set[str] = set()

    def add_lane(name: str) -> None:
        if name not in lane_seen:
            lane_seen.add(name)
            lanes.append(name)

    for participant in run.get("participants") or []:
        if participant.get("name"):
            add_lane(participant["name"])
    add_lane("__aggregator__")

    max_step = 0
    for step in run.get("steps") or []:
        step_start = parse_time(step.get("step_start"))
        step_end = parse_time(step.get("step_end"))
        step_index = step.get("step_index") or 0
        max_step = max(max_step, step_index)

        events.append({
            "id": f"step-{step_index}",
            "lane": "__meeting__",
            "type": "step",
            "label": f"Step {step_index}",
            "x": step_index,
            "x_end": step_index + 1,
            "time": iso(step_start),
            "end": iso(step_end),
            "step": step_index,
            "round": None,
            "detail": step.get("trigger_reason", ""),
            "input": "",
            "output": step.get("trigger_reason", ""),
            "duration_ms": int((step_end - step_start).total_seconds() * 1000)
            if step_start and step_end else None,
        })

        for turn in step.get("turns") or []:
            agent = turn.get("agent") or "__unknown__"
            add_lane(agent)
            round_num = turn.get("round")
            turn_start = parse_time(turn.get("start_time")) or step_start
            turn_end = parse_time(turn.get("end_time")) or step_end
            key = session_key(turn.get("session_path"), agent, round_num)
            session = sessions.get(key or "")

            events.append({
                "id": turn.get("turn_id") or f"{agent}-{round_num}-{step_index}",
                "lane": agent,
                "type": "turn",
                "label": f"{agent} r{round_num}" if round_num else agent,
                "x": step_index + 0.5,
                "x_start": step_index + 0.05,
                "x_end": step_index + 0.95,
                "time": iso(turn_start),
                "end": iso(turn_end),
                "step": step_index,
                "round": round_num,
                "detail": shorten(turn.get("output"), 420),
                "input": turn.get("input"),
                "output": turn.get("output") or "",
                "duration_ms": turn.get("duration_ms"),
                "session": session,
                "role_ref": turn.get("role_ref"),
                "changes": turn.get("changes_from_prior_round"),
            })

            event_items = turn.get("events") or []
            if event_items:
                total_events = len(event_items) + 1
                for event_index, item in enumerate(event_items, start=1):
                    event_type = item.get("type", "event")
                    label = event_type
                    detail = ""
                    input_full: Any = ""
                    output_full: Any = ""
                    if event_type == "llm_call":
                        calls = item.get("response_tool_calls") or []
                        label = "LLM"
                        detail = (
                            f"requested {len(calls)} tool call(s): "
                            + ", ".join(c.get("name", "tool") for c in calls)
                            if calls else "produced text"
                        )
                        if item.get("response_text"):
                            detail += " | " + shorten(item.get("response_text"), 260)
                        input_full = item.get("request_messages")
                        output_full = {
                            "response_text": item.get("response_text"),
                            "response_tool_calls": calls,
                        }
                    elif event_type == "tool_call":
                        label = item.get("tool", "tool")
                        status = "ok" if item.get("success") else "failed"
                        detail = f"{status}; args={shorten(item.get('args'), 180)}"
                        if item.get("result_preview"):
                            detail += f"; result={shorten(item.get('result_preview'), 260)}"
                        input_full = item.get("args")
                        output_full = item.get("result_preview")
                    elif event_type == "compact":
                        label = "compact"
                        detail = item.get("reason", "")
                        output_full = item.get("reason", "")

                    events.append({
                        "id": f"{turn.get('turn_id')}-{item.get('seq')}",
                        "lane": agent,
                        "type": event_type,
                        "label": label,
                        "x": step_index + 0.08 + 0.84 * (event_index / total_events),
                        "time": item.get("start_time") or iso(turn_start),
                        "end": item.get("end_time"),
                        "step": step_index,
                        "round": round_num,
                        "iteration": item.get("iteration"),
                        "duration_ms": item.get("duration_ms"),
                        "detail": detail,
                        "input": input_full,
                        "output": output_full,
                    })
            elif session:
                derived = session.get("derived_events") or []
                total = max(session.get("message_count", len(derived)), 1)
                for item in derived:
                    events.append({
                        "id": f"{key}-{item['message_index']}",
                        "lane": agent,
                        "type": item["type"],
                        "label": item["label"],
                        "x": step_index + 0.08 + 0.84 * ((item["message_index"] + 1) / (total + 1)),
                        "time": iso(turn_start),
                        "end": None,
                        "step": step_index,
                        "round": round_num,
                        "message_index": item["message_index"],
                        "detail": item["detail"],
                        "input": item.get("input", ""),
                        "output": item.get("output", ""),
                    })

    return {
        "meeting": {
            "meeting_id": run.get("meeting_id"),
            "mode": run.get("mode"),
            "status": run.get("status"),
            "created_at": run.get("created_at"),
            "closed_at": run.get("closed_at"),
            "step_count": max_step + 1,
        },
        "lanes": lanes,
        "events": events,
        "sessions": sessions,
    }


def render_html(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    title = f"Session timeline - {data['meeting']['meeting_id']}"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
:root {{
  color-scheme: light dark;
  --bg: #f6f7f9;
  --fg: #17202a;
  --muted: #607080;
  --panel: #ffffff;
  --line: #d8dee8;
  --llm: #2563eb;
  --tool: #059669;
  --turn: #7c3aed;
  --step: #475569;
  --warn: #dc2626;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #0d1117;
    --fg: #e6edf3;
    --muted: #9aa7b4;
    --panel: #161b22;
    --line: #30363d;
  }}
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font: 14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; background: var(--bg); color: var(--fg); }}
header {{ padding: 18px 22px 12px; border-bottom: 1px solid var(--line); background: var(--panel); position: sticky; top: 0; z-index: 5; }}
h1 {{ margin: 0 0 10px; font-size: 20px; font-weight: 650; }}
.summary, .controls {{ display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; }}
.pill {{ border: 1px solid var(--line); border-radius: 999px; padding: 4px 10px; color: var(--muted); background: color-mix(in srgb, var(--panel) 90%, var(--line)); }}
.controls {{ margin-top: 12px; }}
button, select {{ border: 1px solid var(--line); border-radius: 8px; padding: 7px 10px; background: var(--panel); color: var(--fg); }}
main {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(420px, 42vw); gap: 16px; padding: 16px; }}
#chart {{ min-width: 0; overflow-x: auto; background: var(--panel); border: 1px solid var(--line); border-radius: 12px; }}
#detail {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px; position: sticky; top: 118px; align-self: start; max-height: calc(100vh - 136px); overflow: auto; }}
.timeline {{ position: relative; min-width: var(--timeline-width, 1400px); padding: 50px 26px 24px 150px; }}
.axis {{ position: absolute; left: 150px; right: 18px; top: 20px; height: 1px; background: var(--line); }}
.tick {{ position: absolute; top: -7px; width: 1px; height: 15px; background: var(--line); }}
.tick span {{ position: absolute; top: -20px; transform: translateX(-50%); color: var(--muted); font-size: 12px; white-space: nowrap; }}
.lane {{ position: relative; height: 98px; border-top: 1px solid var(--line); }}
.lane-name {{ position: absolute; left: -132px; top: 36px; width: 120px; color: var(--muted); text-align: right; font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.bar {{ position: absolute; top: 38px; height: 22px; border-radius: 999px; opacity: .22; background: var(--turn); }}
.point {{ position: absolute; top: var(--point-top, 34px); width: 24px; height: 24px; border-radius: 999px; transform: translate(-12px, 0); border: 3px solid var(--panel); cursor: pointer; box-shadow: 0 0 0 1px color-mix(in srgb, var(--fg) 18%, transparent); }}
.point.llm_call {{ background: var(--llm); }}
.point.tool_call {{ background: var(--tool); }}
.point.turn {{ background: var(--turn); width: 30px; height: 30px; transform: translate(-15px, -3px); }}
.point.step {{ background: var(--step); }}
.point.compact {{ background: var(--warn); }}
.point.tool_request, .point.tool_result {{ background: var(--tool); }}
.point.assistant_text {{ background: var(--llm); }}
.point.user_message {{ background: var(--step); }}
.point.is-selected {{ outline: 4px solid color-mix(in srgb, var(--fg) 36%, transparent); z-index: 3; }}
.legend {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 0 0 12px; color: var(--muted); }}
.legend span::before {{ content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: -1px; background: var(--step); }}
.legend .llm::before {{ background: var(--llm); }}
.legend .tool::before {{ background: var(--tool); }}
.legend .turn::before {{ background: var(--turn); }}
.detail-title {{ font-weight: 650; font-size: 16px; margin-bottom: 6px; }}
.kv {{ display: grid; grid-template-columns: 110px minmax(0, 1fr); gap: 5px 8px; margin: 12px 0; }}
.kv div:nth-child(odd) {{ color: var(--muted); }}
pre {{ white-space: pre-wrap; word-break: break-word; background: color-mix(in srgb, var(--panel) 86%, var(--line)); padding: 10px; border-radius: 8px; border: 1px solid var(--line); }}
details {{ border-top: 1px solid var(--line); padding-top: 10px; margin-top: 10px; }}
summary {{ cursor: pointer; font-weight: 650; }}
@media (max-width: 900px) {{
  main {{ grid-template-columns: 1fr; }}
  #detail {{ position: static; max-height: none; }}
}}
</style>
</head>
<body>
<header>
  <h1>{escape(title)}</h1>
  <div class="summary" id="summary"></div>
  <div class="controls">
    <select id="roundFilter" aria-label="Round filter"></select>
    <select id="typeFilter" aria-label="Event type filter"></select>
    <button id="reset">Reset selection</button>
  </div>
</header>
<main>
  <section id="chart" aria-label="Meeting timeline"></section>
  <aside id="detail"><div class="detail-title">Select an event</div><p class="pill">Click any mark to inspect the full input and output for that step event.</p></aside>
</main>
<script>
const DATA = {payload};
const chart = document.getElementById('chart');
const detail = document.getElementById('detail');
const roundFilter = document.getElementById('roundFilter');
const typeFilter = document.getElementById('typeFilter');
const reset = document.getElementById('reset');
const stepCount = Math.max(1, DATA.meeting.step_count || 1);
let selectedId = null;

function fmtDur(ms) {{
  if (ms == null) return '';
  if (ms < 1000) return `${{ms}} ms`;
  const sec = ms / 1000;
  if (sec < 90) return `${{sec.toFixed(1)}} s`;
  return `${{(sec / 60).toFixed(1)}} min`;
}}
function escapeHtml(value) {{
  return String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
}}
function fullText(value) {{
  if (value == null || value === '') return '(empty)';
  return typeof value === 'string' ? value : JSON.stringify(value, null, 2);
}}
function block(title, value, open = true) {{
  const attr = open ? ' open' : '';
  return `<details${{attr}}><summary>${{escapeHtml(title)}}</summary><pre>${{escapeHtml(fullText(value))}}</pre></details>`;
}}
function pctX(value) {{
  return (Number(value || 0) / stepCount) * 100;
}}
function eventVisible(e) {{
  const rf = roundFilter.value;
  const tf = typeFilter.value;
  return (rf === 'all' || String(e.round) === rf) && (tf === 'all' || e.type === tf);
}}
function initFilters() {{
  const rounds = [...new Set(DATA.events.map(e => e.round).filter(v => v != null))].sort((a,b) => a-b);
  roundFilter.innerHTML = '<option value="all">All rounds</option>' + rounds.map(r => `<option value="${{r}}">Round ${{r}}</option>`).join('');
  const types = [...new Set(DATA.events.map(e => e.type))].sort();
  typeFilter.innerHTML = '<option value="all">All event types</option>' + types.map(t => `<option value="${{t}}">${{t}}</option>`).join('');
}}
function renderSummary() {{
  const events = DATA.events.length;
  const sessions = Object.keys(DATA.sessions).length;
  const tools = DATA.events.filter(e => e.type === 'tool_call').length;
  document.getElementById('summary').innerHTML = `
    <span class="pill">${{DATA.meeting.status}}</span>
    <span class="pill">${{stepCount}} steps on x-axis</span>
    <span class="pill">${{sessions}} sessions</span>
    <span class="pill">${{events}} events</span>
    <span class="pill">${{tools}} tool calls</span>`;
}}
function renderChart() {{
  const lanes = DATA.lanes.filter(l => l !== '__meeting__');
  const width = Math.max(1400, stepCount * 210 + 180);
  chart.style.setProperty('--timeline-width', `${{width}}px`);
  let html = '<div class="timeline"><div class="legend"><span class="turn">turn</span><span class="llm">llm</span><span class="tool">tool</span><span>step/other</span></div><div class="axis">';
  for (let i = 0; i <= stepCount; i++) {{
    const x = (i / stepCount) * 100;
    html += `<div class="tick" style="left:${{x}}%"><span>Step ${{i}}</span></div>`;
  }}
  html += '</div>';
  for (const lane of lanes) {{
    html += `<div class="lane" data-lane="${{escapeHtml(lane)}}"><div class="lane-name" title="${{escapeHtml(lane)}}">${{escapeHtml(lane)}}</div>`;
    const turnEvents = DATA.events.filter(e => e.lane === lane && e.type === 'turn' && eventVisible(e));
    for (const e of turnEvents) {{
      const left = pctX(e.x_start ?? e.x);
      const right = pctX(e.x_end ?? e.x);
      html += `<div class="bar" style="left:${{left}}%; width:${{Math.max(0.7, right-left)}}%" title="${{escapeHtml(e.label)}}"></div>`;
    }}
    const points = DATA.events.filter(e => e.lane === lane && e.type !== 'step' && eventVisible(e))
      .sort((a, b) => (a.x || 0) - (b.x || 0));
    for (let i = 0; i < points.length; i++) {{
      const e = points[i];
      const top = e.type === 'turn' ? 34 : 14 + (i % 4) * 18;
      html += `<button class="point ${{escapeHtml(e.type)}} ${{e.id === selectedId ? 'is-selected' : ''}}" style="left:${{pctX(e.x)}}%; --point-top:${{top}}px" data-id="${{escapeHtml(e.id)}}" aria-label="${{escapeHtml(e.label)}} at step ${{e.x?.toFixed ? e.x.toFixed(2) : e.x}}"></button>`;
    }}
    html += '</div>';
  }}
  html += '</div>';
  chart.innerHTML = html;
  chart.querySelectorAll('.point').forEach(el => el.addEventListener('click', () => selectEvent(el.dataset.id)));
}}
function selectEvent(id) {{
  selectedId = id;
  const e = DATA.events.find(item => item.id === id);
  if (!e) return;
  const session = e.session;
  detail.innerHTML = `<div class="detail-title">${{escapeHtml(e.label)}}</div>
    <div class="kv">
      <div>Type</div><div>${{escapeHtml(e.type)}}</div>
      <div>Lane</div><div>${{escapeHtml(e.lane)}}</div>
      <div>X</div><div>step ${{e.step ?? 'n/a'}} / position ${{Number(e.x ?? 0).toFixed(3)}}</div>
      <div>Duration</div><div>${{fmtDur(e.duration_ms) || 'n/a'}}</div>
      <div>Step</div><div>${{e.step ?? 'n/a'}}</div>
      <div>Round</div><div>${{e.round ?? 'n/a'}}</div>
      <div>Iteration</div><div>${{e.iteration ?? 'n/a'}}</div>
      ${{e.role_ref ? `<div>Role</div><div>${{escapeHtml(e.role_ref)}}</div>` : ''}}
      ${{session ? `<div>Session</div><div>${{session.message_count}} messages, ${{session.tool_calls}} tool requests, ${{session.tool_results}} tool results</div>` : ''}}
    </div>
    ${{e.changes ? block('Changes from prior round', e.changes, false) : ''}}
    ${{block('Detail', e.detail || '(empty)', false)}}
    ${{block('Full input', e.input, true)}}
    ${{block('Full output', e.output, true)}}
    ${{session ? block('Tools in session', session.tools, false) : ''}}
    ${{session ? block('Full persisted session messages', session.messages, false) : ''}}`;
  renderChart();
}}
function wire() {{
  initFilters();
  renderSummary();
  renderChart();
  roundFilter.addEventListener('change', renderChart);
  typeFilter.addEventListener('change', renderChart);
  reset.addEventListener('click', () => {{ selectedId = null; renderChart(); detail.innerHTML = '<div class="detail-title">Select an event</div><p class="pill">Click any mark to inspect the full input and output for that step event.</p>'; }});
}}
wire();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a meeting session timeline.")
    parser.add_argument("--meeting-id", help="Meeting id under runs/, for example mtg_14bf5d2086")
    parser.add_argument("--run-json", type=Path, help="Explicit path to runs/<meeting_id>.json")
    parser.add_argument("--sessions-dir", type=Path, help="Explicit path to sessions directory")
    parser.add_argument("--output", type=Path, help="Output HTML path")
    args = parser.parse_args()

    if not args.meeting_id and not args.run_json:
        parser.error("provide --meeting-id or --run-json")

    run_json = args.run_json or (RUNS_DIR / f"{args.meeting_id}.json")
    if not run_json.exists():
        raise FileNotFoundError(run_json)

    run = json.loads(run_json.read_text(encoding="utf-8"))
    meeting_id = run.get("meeting_id") or args.meeting_id or run_json.stem
    sessions_dir = args.sessions_dir or (RUNS_DIR / meeting_id / "sessions")
    output = args.output or (RUNS_DIR / meeting_id / "session_timeline.html")
    output.parent.mkdir(parents=True, exist_ok=True)

    sessions = load_sessions(sessions_dir)
    data = build_timeline(run, sessions)
    output.write_text(render_html(data), encoding="utf-8")

    print(f"timeline: {output}")
    print(f"sessions: {len(sessions)}")
    print(f"events: {len(data['events'])}")


if __name__ == "__main__":
    main()
