# Agent-Meeting

A backend for testing multi-agent "meeting" formats: multiple role-agents answer a
question, get merged into one final answer, and every step is logged with enough
granularity to later build a timeline-style visualization on top.

Participant agents are `research_agent.agent.GeneralAgent` instances (from the
[Agent-Tutorial](../Agent-Tutorial) repo, editable-installed as `research-agent`),
running with the full built-in tool registry minus kanban tools.

## Run the example

```powershell
C:\Users\LX034\miniconda3\python.exe examples\run_parallel_qa.py
```

This runs one `parallel_qa` meeting: N participants answer the question concurrently,
then an LLM aggregator synthesizes their answers into `final_response`. Output is saved
to `runs/<meeting_id>.json`.

## Usage

```python
from agent_meeting import MeetingConfig, ParticipantConfig, run_meeting

config = MeetingConfig(
    question="...",
    participants=[
        ParticipantConfig(name="Alice", role="..."),
        ParticipantConfig(name="Bob", role="..."),
    ],
    aggregation_strategy="llm",  # or "concat"
)
result = run_meeting(config)
print(result["final_response"])
```

## Meeting record schema (`runs/<meeting_id>.json`)

Top-level unit is a **step** — one "time unit" / concurrent batch — each holding a list
of **turns** that ran concurrently within it:

```jsonc
{
  "meeting_id": "mtg_...", "mode": "parallel_qa", "question": "...",
  "orchestration": {"type": "scripted"},   // "scripted" | "moderator" | "hybrid"
  "participants": [{"name": "...", "role": "...", "model": "...", "provider": "..."}],
  "steps": [
    {
      "step_index": 0, "step_start": "...", "step_end": "...",
      "decided_by": "script",              // script | moderator | config
      "trigger_reason": "...",
      "turns": [
        {
          "agent": "Alice", "decided_by": "script",
          "available_tools": ["browser_search", "..."],
          "input": [ /* exact messages array sent on the turn's first LLM call */ ],
          "output": "...",                 // final answer text
          "session_id": "...", "session_path": "...",
          "start_time": "...", "end_time": "...", "duration_ms": 5231,
          "events": [                       // full per-iteration trajectory
            {"type": "llm_call", "iteration": 1, "request_messages": [...],
             "response_text": null, "response_tool_calls": [...], "duration_ms": 812},
            {"type": "tool_call", "iteration": 1, "tool": "browser_search",
             "args": {...}, "result_preview": "...", "success": true, "duration_ms": 640}
          ]
        }
      ]
    },
    {"step_index": 1, "decided_by": "config", "trigger_reason": "aggregation", "turns": [...]}
  ],
  "final_response": "..."
}
```

`request_messages` on every `llm_call` event is the **exact** messages array sent to the
model API for that call — not a paraphrase. That matters once a future pass adds
per-agent persistent memory and automatic context compaction: whatever actually got
sent (compacted or not) is what gets logged, with no special-casing needed.

Three orchestration styles fit this same shape without redesign:
- **scripted** — every step/turn has `decided_by: "script"` (what `parallel_qa` does today).
- **moderator** — a step's `decided_by` is `"moderator"`, and that step includes a
  `"__moderator__"` turn recording the moderator's own decision trajectory.
- **hybrid** — a run whose steps mix `decided_by` values.

## Notes

- Each participant's flat message-array session file lives under
  `runs/<meeting_id>/sessions/<name>_r<round>.json` because we pass an explicit
  `session_path=` to `GeneralAgent`. `research_agent/agent.py` was patched (small,
  backward-compatible change: when `session_path` is supplied, it skips its own default
  write to `Agent-Tutorial/sessions/` instead of writing the same messages to both places)
  so this project's `runs/` directory is now the only place a participant's session file
  is written. `usage_log.jsonl` (token counts) still accumulates in
  `Agent-Tutorial/sessions/` — that log is process-wide, not per-session-path, and isn't
  redirected.
- Kanban tools are excluded for participants; every other built-in tool (browser, files,
  terminal, memory, skills, subprocess, self_code, background, respond) is available.
