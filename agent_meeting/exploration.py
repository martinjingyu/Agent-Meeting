"""Pre-roster exploration/confirmation phase: before role_architect.design_domain_roster()
ever runs, one GeneralAgent tool loop reads the task spec and interactively asks a
human stakeholder whatever concrete questions it needs -- via ask_user_question
(exploration_tools.py), rendered as a Claude-Code-style numbered-menu-plus-"Other" TUI
(interactive.ask_choice) -- to resolve real ambiguities before any participant role or
domain brief is designed around a guess.

This mirrors role_architect.py's shape deliberately (same GeneralAgent construction,
same workspace convention, same registry-filtering call) so the two phases -- explore
first, then design the roster -- read as one pipeline rather than two unrelated
mechanisms.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from research_agent.agent import GeneralAgent

from ._context_limits import AUTO_COMPACT, COMPACT_TOKEN_THRESHOLD
from .exploration_tools import register_exploration_tools
from .storage import participant_workspace_dir
from .tools_setup import build_participant_registry
from .trajectory import log

EXPLORATION_MODEL = "gpt-5.5"
EXPLORATION_PROVIDER = "codex"
EXPLORATION_REASONING_EFFORT = "high"
EXPLORATION_MAX_ITERATIONS = 20


@dataclass
class ExplorationResult:
    qa: list[dict] = field(default_factory=list)
    """Every question asked and answered, in order: [{"question", "options", "answer"}]."""
    summary: str = ""
    """The agent's own closing summary (via respond_to_user) of what was confirmed --
    meant to be dropped into the task spec handed to design_domain_roster()."""

    def render_task_spec_addendum(self) -> str:
        if not self.qa:
            return ""
        lines = [
            "=== Human clarification (asked and answered before the roster was designed) ===",
            "",
            "The following questions were asked directly to the human stakeholder during "
            "an exploration phase before any participant role existed. Every answer below "
            "is authoritative human input, not a model guess -- treat it as a hard "
            "constraint on the roster and the meeting, not one opinion among several.",
            "",
        ]
        for i, entry in enumerate(self.qa, 1):
            lines.append(f"{i}. Q: {entry['question']}")
            lines.append(f"   A: {entry['answer']}")
        if self.summary:
            lines.append("")
            lines.append(f"Confirmed understanding: {self.summary}")
        return "\n".join(lines)


def _build_prompt(task_spec: str, min_questions: int, max_questions: int) -> tuple[str, str]:
    system_prompt = (
        "You are running the exploration phase of a technical planning pipeline, "
        "before any participant role or meeting has been designed. Your only job "
        "is to remove real ambiguity from the task spec below by asking the human "
        "stakeholder directly -- you do not propose a solution, a roster, or an "
        "architecture here.\n\n"
        "Read the task spec carefully and identify the points where two competent "
        "engineers could reasonably build materially different things from it: an "
        "unstated priority between competing goals, a scope boundary left implicit, "
        "a constraint that's plausible but not actually stated, a tradeoff the spec "
        "gestures at without picking a side. For each one you find, call "
        "ask_user_question with a single specific question and 2-4 concrete candidate "
        "answers -- write it so a non-expert stakeholder can pick one without having "
        "to write anything (a free-text 'Other' option is always offered "
        "automatically). Ask one question per call, and read each answer before "
        "deciding your next question -- an earlier answer can make a question you "
        "were about to ask irrelevant, or reveal a new one.\n\n"
        f"Ask at least {min_questions} question(s) if the spec leaves that many "
        f"genuine ambiguities open, and no more than {max_questions} -- do not pad "
        "with a question the spec already answers just to hit a minimum, and do not "
        "stop early while a real ambiguity that would change the roster or the "
        "discussion's focus is still open.\n\n"
        "When you're done, call respond_to_user with a short summary (a few "
        "sentences) of the confirmed understanding -- what was clarified and how it "
        "constrains the work to come. This summary, together with the full Q&A log, "
        "is handed directly to whatever designs the meeting's participant roster."
    )
    user_message = f"=== Task spec ===\n{task_spec}\n\nBegin the exploration phase now."
    return system_prompt, user_message


def run_exploration_phase(
    task_spec: str,
    meeting_id: str,
    *,
    min_questions: int = 1,
    max_questions: int = 6,
    model: str = EXPLORATION_MODEL,
    provider: str = EXPLORATION_PROVIDER,
    reasoning_effort: str = EXPLORATION_REASONING_EFFORT,
    max_iterations: int = EXPLORATION_MAX_ITERATIONS,
    verbose: bool = True,
) -> ExplorationResult:
    """Blocks on real terminal input for however many questions the agent asks --
    meant to run once, interactively, before the meeting's roster is designed. Like
    design_domain_roster(), meeting_id is required even though this runs before the
    meeting proper, so the agent gets a real scoped workspace_root."""
    register_exploration_tools()
    system_prompt, user_message = _build_prompt(task_spec, min_questions, max_questions)

    if verbose:
        log("exploration", "starting pre-roster exploration -- may ask clarifying questions...")

    registry = build_participant_registry(role_backed=False, round_aware=False)
    agent = GeneralAgent(
        model=model,
        provider=provider,
        reasoning_effort=reasoning_effort,
        max_iterations=max_iterations,
        context_threshold_tokens=COMPACT_TOKEN_THRESHOLD,
        auto_compact=AUTO_COMPACT,
        self_review=False,
        registry=registry,
        sub_agent=True,
        agent_role="exploration",
        workspace_root=participant_workspace_dir(meeting_id, "Exploration"),
    )
    agent.run(user_message, system_prompt=system_prompt)
    runtime = getattr(agent, "_runtime", {})
    qa: list[dict] = runtime.get("clarification_qa") or []
    summary = str(runtime.get("final_response") or "")

    if verbose:
        log("exploration", f"finished -- {len(qa)} question(s) asked and answered")
    return ExplorationResult(qa=qa, summary=summary)
