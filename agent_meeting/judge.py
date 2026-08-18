"""Judge for mode="planning_rounds": after each ideation round, decides whether the
discussion has surfaced enough for the Planner to synthesize a Plan, or whether
another round would meaningfully add something.

Two implementations, chosen by MeetingConfig.human_checkin:

- False (the default): judge_should_stop(), a single deterministic LLM call (same
  pattern as aggregate.py's _aggregate_llm), not a GeneralAgent turn -- the judge
  only ever reads a transcript and answers a strict stop/continue, it never needs
  tools. Always Codex gpt-5.5 -- an internal gatekeeping decision, not something a
  caller tunes per meeting the way aggregation_model/provider are.

- True: run_interactive_judge(), a GeneralAgent tool loop with exactly two tools
  (judge_tools.build_judge_registry: ask_user_question, submit_judgment) instead of
  one fixed completion. This lets the judge have a genuine multi-turn conversation
  with the human stakeholder -- ask a question, get an answer, ask a follow-up if
  the answer shows they didn't understand, repeat for as many exchanges as it takes
  -- before finishing with its stop/continue verdict, rather than a single Q+A
  bolted on after a verdict already reached without human input. See runner.py's
  _run_judge_step for how the two are dispatched and how run_interactive_judge's
  qa log gets folded into the round's transcript.

Both must emit their reasoning (open issues weighed, per-participant coverage)
before the verdict, not a bare "1"/"0". Two reasons: (1) auditability -- a bare
digit gives no way to check after the fact whether a stop was justified; (2)
quality -- complete_text() only ever returns the model's final visible content,
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

from research_agent.agent import GeneralAgent
from research_agent.llm import LLMClient

from ._context_limits import AUTO_COMPACT, COMPACT_TOKEN_THRESHOLD
from .judge_tools import build_judge_registry
from .storage import participant_workspace_dir

JUDGE_MODEL = "gpt-5.5"
JUDGE_PROVIDER = "codex"
JUDGE_REASONING_EFFORT = "medium"

INTERACTIVE_JUDGE_MODEL = "gpt-5.5"
INTERACTIVE_JUDGE_PROVIDER = "codex"
INTERACTIVE_JUDGE_REASONING_EFFORT = "medium"
INTERACTIVE_JUDGE_MAX_ITERATIONS = 12
"""Well above what a normal exchange needs (each back-and-forth is 2 iterations: one
ask_user_question call, one model turn reading the answer) -- generous rather than
tight, since a genuinely confused human working through 3-4 clarifying rounds before
submit_judgment is exactly the scenario this mode exists for, not a runaway loop to
guard against as tightly as a participant's own turn budget."""

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


def _round_budget_note(round_num: int | None, max_rounds: int | None) -> str:
    """Shared between judge_should_stop's prompt and run_interactive_judge's system
    prompt -- both need the model to weigh how many rounds are actually left the same
    way, so this stays as one piece of text rather than drifting apart across the two
    implementations."""
    rounds_left = None
    if round_num is not None and max_rounds is not None:
        rounds_left = max_rounds - round_num
    if rounds_left is not None and rounds_left <= 0:
        return (
            f"\n=== Round budget ===\nThis was round {round_num} of {max_rounds} -- the "
            "meeting's round budget is exhausted; there is no further round available "
            "regardless of what you decide here. A stop=false verdict will NOT produce "
            "another round of discussion; it only tells the Planner that open gaps "
            "remain. So: do not simply say 'continue' and leave issues open the way you "
            "would if another round were coming. Instead, for every genuine unresolved "
            "issue you would otherwise defer to a future round, resolve it yourself as "
            "far as the existing transcript allows (state which participant's position "
            "you find more supported and why) so unresolved_issues becomes concrete "
            "guidance the Planner can act on directly, not an open question with no one "
            "left to answer it. Set stop=true if the remaining gaps are now reduced to "
            "Planner-resolvable choices this way.\n"
        )
    if rounds_left is not None:
        return (
            f"\n=== Round budget ===\nThis was round {round_num} of {max_rounds} -- "
            f"{rounds_left} round(s) remain after this one. Weigh that when deciding: an "
            "issue worth another full round of participant interaction is still "
            "stop=false, but if very few rounds remain, prefer flagging issues in a form "
            "specific enough that participants can close them quickly (name the exact "
            "test or reconciliation needed) rather than leaving them open-ended.\n"
        )
    return ""


def judge_should_stop(
    question: str,
    transcript: str,
    *,
    round_num: int | None = None,
    max_rounds: int | None = None,
) -> dict[str, Any]:
    budget_note = _round_budget_note(round_num, max_rounds)

    prompt = (
        "You are the gatekeeper for a planning discussion. Below is the task and every "
        "round of participant contributions so far (points, suggestions, and ideas only "
        "-- no Plan has been written yet).\n\n"
        f"=== Task ===\n{question}\n\n"
        f"=== Discussion So Far ===\n{transcript}\n\n"
        f"{budget_note}"
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


def run_interactive_judge(
    question: str,
    transcript: str,
    meeting_id: str,
    *,
    round_num: int | None = None,
    max_rounds: int | None = None,
    progress: Any = None,
    model: str = INTERACTIVE_JUDGE_MODEL,
    provider: str = INTERACTIVE_JUDGE_PROVIDER,
    reasoning_effort: str = INTERACTIVE_JUDGE_REASONING_EFFORT,
    max_iterations: int = INTERACTIVE_JUDGE_MAX_ITERATIONS,
    verbose: bool = True,
) -> dict[str, Any]:
    """MeetingConfig.human_checkin's judge. progress, if given, is threaded into the
    agent's runtime as "judge_meeting_progress" -- judge_tools.py's ask_user_question
    handler pauses it (a duck-typed .paused() context manager) around its blocking
    input() prompt, the same way exploration_tools.py does; passed through untyped
    (Any) here so this module doesn't need to import agent_meeting.tui just for a
    type hint.

    Returns the same stop/reasoning/unresolved_issues/per_participant_coverage shape
    as judge_should_stop(), plus "qa": the full ask_user_question back-and-forth log,
    in order, how ever many exchanges it took -- runner.py's _run_judge_step folds
    each entry into the round's transcript as an authoritative human turn."""
    budget_note = _round_budget_note(round_num, max_rounds)
    system_prompt = (
        "You are the gatekeeper for a multi-round technical planning discussion. "
        "Below is the task and every round of participant contributions so far "
        "(points, suggestions, and ideas only -- no Plan has been written yet).\n\n"
        f"{budget_note}\n"
        "Your job is NOT to decide whether a Planner could write some plan now. Your "
        "job is to decide whether another participant discussion round would still "
        "create meaningful value before the Planner writes the final Plan.\n\n"
        "Set stop=false when another round is likely to resolve, test, or sharpen a "
        "material issue through participant interaction: unresolved disagreement "
        "between participants, an objection that has not been answered by the "
        "relevant owner, conflicting empirical claims, a proposed pipeline step whose "
        "interface or acceptance criteria remain unclear, or a failure mode that "
        "needs a concrete mitigation rather than being handed to the Planner as a "
        "TODO.\n\n"
        "Set stop=true only when further discussion would mostly repeat known "
        "positions or add minor implementation polish, and the remaining choices are "
        "clearly Planner synthesis decisions rather than open discussion gaps.\n\n"
        "Pay particular attention to any participant who raised an explicit "
        "unresolved objection, dissent, or a 'REVISE'/'REJECT' verdict of their own "
        "that the same participant hasn't since withdrawn -- that is a blocking "
        "discussion gap, not merely a Planner decision issue, and belongs in "
        "unresolved_issues.\n\n"
        "Also watch for a specific quantitative claim or threshold that multiple "
        "participants have repeated as settled fact across rounds without anyone "
        "ever actually running a test that measures it -- list it in "
        "unresolved_issues (phrased as 'X has been assumed but never measured') and "
        "let it count toward stop=false, even though no participant currently "
        "disagrees with it.\n\n"
        "You have two tools. ask_user_question talks directly to the human "
        "stakeholder who commissioned this task -- not a meeting participant, a real "
        "person who has not read the discussion below. You may call it any number of "
        "times, back and forth: if their answer shows they didn't understand your "
        "question, or is itself a question, clarify and ask again rather than "
        "guessing at what they meant -- keep going until you have a real, usable "
        "answer. Before you finish, ask about this round's single most consequential "
        "open point -- a direction choice between two approaches participants both "
        "defended, a priority tradeoff no one in the discussion has the authority to "
        "settle, a scope boundary the discussion exposed as ambiguous, or a risk "
        "flagged without a clear owner -- phrased so a non-expert stakeholder can "
        "answer it directly, with 2-4 concrete candidate answers. Skip asking only "
        "when you're genuinely confident nothing this round needs their input. "
        "submit_judgment ends your turn with your final verdict -- call it once "
        "you've asked everything you need to (including zero questions if none were "
        "warranted).\n\n"
        f"=== Task ===\n{question}\n\n"
        f"=== Discussion So Far ===\n{transcript}"
    )
    user_message = "Review this round and decide -- ask the human anything you need to before submit_judgment."

    if verbose:
        from .trajectory import log
        log("judge", f"round {round_num}: reviewing -- may ask you a question...")

    agent = GeneralAgent(
        model=model,
        provider=provider,
        reasoning_effort=reasoning_effort,
        max_iterations=max_iterations,
        context_threshold_tokens=COMPACT_TOKEN_THRESHOLD,
        auto_compact=AUTO_COMPACT,
        self_review=False,
        registry=build_judge_registry(),
        sub_agent=True,
        agent_role="judge",
        workspace_root=participant_workspace_dir(meeting_id, "Judge"),
        extra_runtime={"judge_meeting_progress": progress},
    )
    agent.run(user_message, system_prompt=system_prompt)
    runtime = getattr(agent, "_runtime", {})
    judgment: dict[str, Any] | None = runtime.get("judgment")
    qa: list[dict[str, Any]] = runtime.get("judge_qa") or []

    if judgment is None:
        # Fail open (continue), same reasoning as judge_should_stop's parse-failure
        # path -- the agent ran out of iterations or otherwise never called
        # submit_judgment; burning one extra round is cheap, losing the rest of a
        # planning meeting to a judge that never decided is not.
        return {
            "stop": False,
            "reasoning": "Interactive judge finished without calling submit_judgment; defaulting to continue.",
            "unresolved_issues": ["judge_output_parse_failure"],
            "per_participant_coverage": {},
            "qa": qa,
        }
    return {
        "stop": bool(judgment.get("stop")),
        "reasoning": str(judgment.get("reasoning") or ""),
        "unresolved_issues": judgment.get("unresolved_issues") or [],
        "per_participant_coverage": judgment.get("per_participant_coverage") or {},
        "qa": qa,
    }
