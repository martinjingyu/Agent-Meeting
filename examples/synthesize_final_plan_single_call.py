"""Synthesizes a completed meeting's final Plan with a SINGLE LLM call instead of
the framework's normal Planner step (a multi-iteration GeneralAgent tool loop that
reads shared/ artifacts, writes the plan to its own workspace, etc. -- see
agent_meeting/runner.py's _run_planner_step). This script does none of that: it
loads the saved meeting checkpoint, assembles the task + full round-by-round
transcript into one request, and calls gpt-5.6-sol once, with no tools.

Critical design constraint, not a nice-to-have: the Executor who eventually
implements this plan will read ONLY the saved .md file. They will never open the
meeting transcript, never see a participant's scripts or shared/<name>/ artifacts,
never know which participant argued what in which round. So SYSTEM_PROMPT below
explicitly demands the plan be self-contained -- every decision's reasoning and
every value it depends on must be inlined, and every bit of meeting-internal
shorthand (participant names, round numbers, "as X argued", ad hoc names for
techniques/components/artifacts that only made sense inside the meeting) must be
translated into plain engineering language an Executor with zero meeting context
can act on. "The relation types from the discussion" is not acceptable in the
output; "duplicate detection using perceptual hashing" is.

Run:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\synthesize_final_plan_single_call.py mtg_xxxxxxxxxx

Optional: write to a specific path instead of the default
runs/<meeting_id>_single_call_plan.md:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\synthesize_final_plan_single_call.py mtg_xxxxxxxxxx --out C:\\path\\to\\plan.md
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Reused rather than reimplemented -- this is the exact function the framework's own
# agentic Planner step uses to flatten a meeting's round-by-round turns into one
# transcript (agent_meeting/runner.py). Keeping this in sync with that function
# means both synthesis paths see the discussion in the same shape, so any
# difference in their output is due to the synthesis approach, not to a
# transcript-formatting discrepancy between the two scripts.
from agent_meeting.runner import _round_transcript
from agent_meeting.single_call_synthesis import stream_synthesize
from agent_meeting.storage import load_meeting, meeting_path

MODEL = "deepseek-v4-pro"
PROVIDER = "deepseek"
REASONING_EFFORT = "high"


SYSTEM_PROMPT = """\
You are synthesizing the final Plan from a completed multi-round technical planning
meeting. Participants (organized by technical domain) spent several rounds
contributing free-form viewpoints, running small experiments, and correcting each
other -- none of them drafted a final Plan themselves. That synthesis is your job,
in a single pass: read the task and the entire discussion below, resolve every
disagreement explicitly (state which position you are adopting and why, not both
positions side by side), and produce one coherent, executable plan.

=== The single most important constraint on this document ===

The Executor who implements this plan will read ONLY the document you produce right
now. They will never see the meeting transcript below, never see any script or file
a participant wrote during the meeting, never know this meeting happened at all.
That has concrete consequences for how you must write:

- Resolve every meeting-internal reference before it reaches the output. Do not
  write "the relation types discussed", "as argued in an earlier round", "per
  ClassicalVision's proposal", "see shared/<participant>/report.md", or any other
  phrasing that only makes sense to someone who read the meeting. State the actual
  content directly and completely, as if you derived it yourself.
- Translate ad hoc terminology into standard engineering language. Participants
  sometimes invent a shorthand name for a technique, a data structure, or a
  category that made sense in the flow of discussion but is not a term the
  Executor will recognize cold. Either replace it with the standard name for what
  it actually is (e.g. a participant's "diff-locality classification" is pixel
  registration / alignment-based duplicate confirmation -- name it that way) or,
  if no standard term fits, define it plainly on first use and then use that
  definition consistently.
- Completeness over brevity. Never defer a detail to "see the discussion" or "as
  established earlier" -- if a value, threshold, formula, or decision is needed to
  implement the plan, it must be written in this document, in the appendix if it's
  dense reference material, but written. An Executor who has only this file must be
  able to build the entire system without asking a follow-up question about what a
  term means or where a number came from.
- Do not name participants, round numbers, or the meeting process itself anywhere
  in the output. The document should read as if a single engineer designed the
  whole thing, not as a summary of who said what.

=== Writing style ===

Write like a senior engineer explaining this design to a colleague, not like you
are filling out a spec template:

- Lead with reasoning, not just conclusions. Don't just state what was chosen --
  show the thinking that got there: what the obvious first approach would be, why
  it falls short given the evidence or constraints discussed in the meeting, and
  what that implies about the approach you're adopting instead.
- Write connected prose, not a list of disconnected facts stitched together with
  semicolons. Use bullet points only for content that is genuinely list-shaped.
- Explain each technique in a real clause (what it's for, why it was adopted or
  rejected), not as a bare comma/slash-separated term dump.
- Push genuinely dense reference material (exact thresholds, formulas, schemas,
  directory layouts) into an appendix so the main sections stay readable as an
  argument, not a lookup table -- but the appendix holds the *values*, the main
  section still owns the *why*.

=== Content requirements ===

Do not mechanically include every technique the discussion touched on. For each
important technical choice, distinguish: the adopted primary method; a fallback or
optional method, if any; anything explicitly rejected and why; and any capability
limit that remains open and cannot be resolved by more design (only by future work).

Do not default to a simple linear pipeline merely because it is easy to write --
adopt or reject each major direction based on the evidence and constraints actually
discussed in the meeting. If the meeting left a material disagreement or an
unfinished experiment genuinely unresolved, say so explicitly as a known limitation
rather than silently picking a side without justification or hiding the gap.

Structure the document for readability before implementation detail:

1. executive summary and the selected overall approach;
2. a compact architecture/data-flow diagram (plain text/ASCII is fine);
3. important alternatives that were considered and the explicit decision on each;
4. stage-by-stage (or component-by-component) inputs, outputs, purpose, and
   failure/edge-case behavior;
5. how the design evaluates its own correctness/quality (tests, acceptance
   criteria, proxy metrics) given whatever ground-truth limitations the meeting
   discussed;
6. cost, reliability, caching/recovery, and degradation behavior, if relevant to
   this task;
7. implementation milestones, in an order that lets each one be validated before
   the next depends on it;
8. known limits, risks, and anything left genuinely unresolved by the meeting;
9. an appendix for exact thresholds, formulas, schemas, field names, and directory
   layouts -- detailed enough that no implementation-relevant number is missing
   from the document, without letting that density crowd the main sections above.

The plan must be detailed enough for an engineer with no other context to
implement it without making new methodology-level decisions of their own.

Respond with ONLY the final plan itself, formatted as Markdown, starting with a
top-level heading. No preamble, no meta-commentary about the meeting or your own
process.
"""


def _build_user_message(meeting_id: str) -> tuple[str, dict]:
    checkpoint = load_meeting(meeting_id)
    if checkpoint.get("mode") != "planning_rounds":
        raise SystemExit(
            f"{meeting_id} has mode={checkpoint.get('mode')!r}; this script only "
            "handles planning_rounds meetings (the ones with a round-by-round "
            "participant discussion to synthesize)."
        )

    all_rounds_turns = [
        step["turns"] for step in checkpoint.get("steps", []) if step.get("decided_by") == "script"
    ]
    if not all_rounds_turns:
        raise SystemExit(f"{meeting_id} has no participant discussion rounds to synthesize from.")

    transcript = _round_transcript(all_rounds_turns)
    user_message = (
        f"=== Task ===\n{checkpoint['question']}\n\n"
        f"=== Full Discussion ({len(all_rounds_turns)} round(s), all participants) ===\n"
        f"{transcript}\n\n"
        "Write the final Plan now."
    )
    return user_message, checkpoint


def synthesize_plan(meeting_id: str) -> str:
    user_message, _checkpoint = _build_user_message(meeting_id)
    try:
        return stream_synthesize(
            SYSTEM_PROMPT,
            user_message,
            model=MODEL,
            provider=PROVIDER,
            reasoning_effort=REASONING_EFFORT,
            verbose=True,
            label="synthesize",
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit("usage: synthesize_final_plan_single_call.py <meeting_id> [--out <path>]")
    meeting_id = args[0]

    out_path: Path
    if "--out" in sys.argv:
        out_path = Path(sys.argv[sys.argv.index("--out") + 1])
    else:
        out_path = meeting_path(meeting_id).with_name(f"{meeting_id}_single_call_plan.md")

    print(f"[synthesize] {meeting_id}: reading checkpoint and calling {MODEL} ({PROVIDER}) once...")
    plan = synthesize_plan(meeting_id)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(plan, encoding="utf-8")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
