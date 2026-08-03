"""Judge for mode="planning_rounds": after each ideation round, decides whether the
discussion has surfaced enough for the Planner to synthesize a Plan, or whether
another round would meaningfully add something.

A single deterministic LLM call (same pattern as aggregate.py's _aggregate_llm),
not a GeneralAgent turn -- the judge only ever reads a transcript and answers a
strict stop/continue, it never needs tools. Always Codex gpt-5.5 -- this is an
internal gatekeeping decision, not something a caller tunes per meeting the way
aggregation_model/provider are.

The judge must emit its reasoning (open issues weighed, per-participant coverage)
before the verdict, as JSON -- not a bare "1"/"0". Two reasons: (1) auditability --
a bare digit gives no way to check after the fact whether a stop was justified;
(2) quality -- complete_text() only ever returns the model's final visible content,
discarding any hidden thinking-mode trace, so if the verdict is the only visible
token the "reasoning" the reasoning_effort pays for never actually informs
anything beyond what's already baked into token probabilities. Making the model
write the reasoning out as content forces it to actually condition the verdict on
an explicit accounting of what participants raised, including any unresolved
dissent (e.g. a participant's own "REVISE" verdict) that a bare stop/continue call
could otherwise silently override.
"""
from __future__ import annotations

import json
import re
from typing import Any

from research_agent.llm import LLMClient

JUDGE_MODEL = "gpt-5.5"
JUDGE_PROVIDER = "codex"
JUDGE_REASONING_EFFORT = "medium"

# Verdict tokens (from a role's output_contract enum, e.g. skeptic-reviewer's
# "ACCEPT | REVISE | REJECT") that must force the round loop to continue,
# regardless of what the judge LLM itself concludes. This is a deterministic
# backstop, not a suggestion to the judge -- a free-text "unresolved_issues"
# list is something an LLM judge can rationalize past even when explicitly told
# to weigh dissent heavily (observed in practice: the judge acknowledged
# Skeptic's REVISE verdict in per_participant_coverage, then still reasoned
# "the disagreement was already resolved by Skeptic's own tests" and stopped
# anyway). A structured verdict the judge cannot talk its way around closes
# that loophole. See agent_meeting.roles.extract_output_contract_verdict for
# how a turn's contract_verdict field gets populated.
BLOCKING_VERDICTS = {"REVISE", "REJECT"}

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def blocking_verdicts_this_round(turns: list[dict[str, Any]]) -> dict[str, str]:
    """agent name -> contract_verdict, for any turn this round whose contract_verdict
    is in BLOCKING_VERDICTS. Empty dict means no participant raised one."""
    return {
        turn["agent"]: turn["contract_verdict"]
        for turn in turns
        if turn.get("contract_verdict") in BLOCKING_VERDICTS
    }


def _parse_judge_output(raw: str) -> dict[str, Any] | None:
    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "stop" not in data:
        return None
    return data


def judge_should_stop(question: str, transcript: str) -> dict[str, Any]:
    prompt = (
        "You are the gatekeeper for a planning discussion. Below is the task and every "
        "round of participant contributions so far (points, suggestions, and ideas only "
        "-- no Plan has been written yet).\n\n"
        f"=== Task ===\n{question}\n\n"
        f"=== Discussion So Far ===\n{transcript}\n\n"
        "Your job is NOT to decide whether a Planner could write some plan now. Your "
        "job is to decide whether another participant discussion round would still "
        "create meaningful value before the Planner writes the final Plan.\n\n"
        "Set stop=false when another round is likely to resolve, test, or sharpen a "
        "material issue through participant interaction: unresolved disagreement "
        "between participants, an objection that has not been answered by the relevant "
        "owner, conflicting empirical claims, a proposed pipeline step whose interface "
        "or acceptance criteria remain unclear, or a failure mode that needs a concrete "
        "mitigation rather than being handed to the Planner as a TODO. In these cases, "
        "the right question is: can the next round make the eventual Plan better by "
        "forcing participants to converge or expose the tradeoff?\n\n"
        "Set stop=true only when further discussion would mostly repeat known positions "
        "or add minor implementation polish, and the remaining choices are clearly "
        "Planner synthesis decisions rather than open discussion gaps. A Planner "
        "decision issue is something where the options and tradeoffs are already on "
        "the table; a blocking discussion gap is something participants still need to "
        "answer, reconcile, or falsify.\n\n"
        "Pay particular attention to any participant who raised an explicit unresolved "
        "objection, dissent, or a 'REVISE'/'REJECT' verdict of their own. If the latest "
        "round contains an explicit participant verdict of REVISE or REJECT, set "
        "stop=false unless that same latest-round output explicitly withdraws it or "
        "changes its final verdict to ACCEPT. Do not treat a REVISE/REJECT as merely a "
        "Planner decision issue just because the objection is well-described; an "
        "unwithdrawn REVISE/REJECT is a blocking discussion gap and belongs in "
        "unresolved_issues.\n\n"
        "Also watch for a quieter failure mode: a specific quantitative claim or "
        "threshold (e.g. an error rate, a false-positive percentage, a confidence "
        "number) that multiple participants have repeated as settled fact across two "
        "or more rounds without anyone ever having actually run a test that measures "
        "it -- as opposed to a claim that has already been verified by a concrete "
        "script/sample/full-pass this round or an earlier one. This looks like "
        "agreement, not disagreement, so it is easy to miss; but an untested repeated "
        "number is exactly the kind of thing a further round should force someone to "
        "verify before the Planner builds on it. If you spot one, list it in "
        "unresolved_issues (phrase it as 'X has been assumed but never measured -- "
        "someone should verify it') and let it count toward stop=false, even though no "
        "participant currently disagrees with it.\n\n"
        "Respond with ONLY a JSON object (no markdown fences, no extra text) with "
        "exactly these keys, in this order:\n"
        '  "per_participant_coverage": a short object mapping each participant name to '
        "one sentence on whether their contribution this round was substantive or a "
        "placeholder/no-op,\n"
        '  "unresolved_issues": a list of strings, each a blocking discussion gap '
        "that another participant round could materially improve; do not list mere "
        "Planner synthesis choices here (empty list if genuinely none),\n"
        '  "reasoning": a few sentences explaining the stop/continue call, referencing '
        "unresolved_issues above and explicitly separating discussion gaps from Planner "
        "synthesis decisions,\n"
        '  "stop": true if the meeting should stop now, false if it should continue for '
        "another round."
    )
    llm = LLMClient(model=JUDGE_MODEL, provider=JUDGE_PROVIDER, reasoning_effort=JUDGE_REASONING_EFFORT)
    output = llm.complete_text(prompt).strip()
    parsed = _parse_judge_output(output)
    if parsed is None:
        # Fail open (continue) rather than silently truncating the discussion on a
        # plumbing/parsing failure -- burning one extra round is cheap, losing the
        # rest of a planning meeting to an unparseable judge response is not.
        return {
            "prompt": prompt,
            "output": output,
            "stop": False,
            "reasoning": "Judge output could not be parsed as JSON; defaulting to continue.",
            "unresolved_issues": ["judge_output_parse_failure"],
            "per_participant_coverage": {},
        }
    return {
        "prompt": prompt,
        "output": output,
        "stop": bool(parsed.get("stop")),
        "reasoning": str(parsed.get("reasoning") or ""),
        "unresolved_issues": parsed.get("unresolved_issues") or [],
        "per_participant_coverage": parsed.get("per_participant_coverage") or {},
    }
