from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParticipantConfig:
    name: str
    role: str = ""
    skills: str = ""
    system_prompt: str | None = None
    meeting_brief: str | None = None
    """Optional meeting-specific assignment appended after either the reusable
    role_ref prompt or the ad-hoc participant prompt. This keeps stable role identity
    separate from the concrete responsibility in one meeting."""
    model: str | None = None
    provider: str | None = None
    reasoning_effort: str | None = None
    max_iterations: int = 8
    role_ref: str | None = None
    """Name of a stored role (agent_meeting.roles) to pull in instead of the ad-hoc
    fields above. When set, name/role/skills/system_prompt are ignored -- the role's
    own DEFINITION.md + persistent memory drive the participant's identity, and
    model/provider/max_iterations fall back to the role's own frontmatter defaults
    when left unset here."""
    vision_capable: bool = False
    """Explicit opt-in for the view_image tool -- False (the default) for every
    participant unless set True here. Not inferred from provider: even though only
    provider="codex" can actually deliver image content today (see llm.py's
    _codex_chat / _to_responses_input), this stays a separate manual switch so
    enabling vision for a participant is always a deliberate per-participant choice
    in the meeting script, not something that silently follows from picking a model.
    Setting this True on a non-codex participant raises at turn-build time (see
    runner.py._execute_turn) rather than deferring to a confusing provider-side 400
    the first time that participant's own next call resends the image."""

    def build_system_prompt(self) -> str:
        if self.system_prompt:
            return self.system_prompt
        parts = [f"You are {self.name}, a participant in a multi-agent meeting."]
        if self.role:
            parts.append(f"Role: {self.role}.")
        if self.skills:
            parts.append(f"Skills and knowledge:\n{self.skills}")
        parts.append(
            "Answer the meeting question directly and thoroughly, using tools if they "
            "genuinely help you answer better."
        )
        return "\n\n".join(parts)


@dataclass
class ModeratorConfig:
    name: str = "Moderator"
    role_ref: str | None = None
    """The moderator can itself be a stored role (agent_meeting.roles), same as a
    participant -- if set, system_prompt/model/provider below are ignored in favor of
    the role's own definition/defaults, mirroring ParticipantConfig.role_ref."""
    system_prompt: str | None = None
    model: str | None = None
    provider: str | None = None
    max_iterations: int = 40
    """Higher than a participant's default -- the moderator has to loop through
    add-participant/call-on/notes/conclude across the whole meeting, not just answer
    one question."""


@dataclass
class PlannerConfig:
    name: str = "Planner"
    system_prompt: str | None = None
    model: str | None = None
    provider: str | None = None
    reasoning_effort: str | None = None
    max_iterations: int = 20
    """Higher than a participant's default -- unlike participants (prose-only,
    ideas/suggestions), the planner actually writes the final Plan to disk via file
    tools, which can take a few tool-call iterations."""
    vision_capable: bool = False
    """Same explicit opt-in as ParticipantConfig.vision_capable -- see there."""


@dataclass
class MeetingConfig:
    question: str
    participants: list[ParticipantConfig] = field(default_factory=list)
    mode: str = "parallel_qa"
    aggregation_strategy: str = "llm"
    aggregation_model: str | None = None
    aggregation_provider: str | None = None
    final_audit: bool = False
    """When True, run one final constraint-and-evidence audit over the last
    aggregated answer. This is useful for planning meetings where consensus can
    accidentally amplify an invalid assumption."""
    rounds: int = 1
    verbose: bool = True
    """Print progress lines to stdout while the meeting runs (per-participant
    iteration/tool-call updates). Set False for silent runs."""
    moderator: ModeratorConfig | None = None
    """Required when mode="moderator". participants (if non-empty) pre-seed the
    roster before the moderator starts -- it can still add more dynamically."""
    max_rounds: int = 8
    """mode="planning_rounds" only: hard cap on ideation rounds, independent of the
    judge's stop/continue decision -- a judge that never says stop can't run the
    round loop away indefinitely."""
    planner: PlannerConfig | None = None
    """Required when mode="planning_rounds". Runs once, after the judge stops the
    round loop (or max_rounds is hit), to synthesize the entire multi-round
    discussion into the final Plan -- participants in this mode never draft a Plan
    themselves, so this is the only place the Plan actually gets written."""
    planner_inline_rounds: int | None = 3
    """mode="planning_rounds" only: number of most-recent discussion rounds embedded
    verbatim in the Planner's initial message. Set to None to embed every round.
    When earlier rounds are not embedded, the complete transcript is still written
    to the meeting shared directory and exposed to the Planner as a readable file."""
    planning_participant_addendum: str | None = None
    """mode="planning_rounds" only: optional replacement for the default participant
    system addendum that limits participants to ideas/suggestions. Examples can use
    this to allow candidate architecture sketches or technical data flows while still
    reserving the adopted final Plan for the dedicated Planner."""
    persist_role_state: bool = False
    """Role-backed participants/moderator normally read/write roles/<role_ref>/
    {memory.md,workspace/} -- state that's shared across EVERY meeting that ever
    uses that role_ref (see roles.py). Left False (the default), this meeting
    snapshots each role_ref's memory.md/workspace/ before it starts and rolls them
    back to that exact snapshot once the meeting reaches its final "completed"
    status, so the role ends up looking like the meeting never happened. Set True
    to let this meeting's role_memory writes and workspace files persist into
    future meetings, as they always did before this flag existed. Resuming a
    failed run does not roll anything back mid-flight -- only a meeting that
    actually finishes triggers the rollback. See role_state.py."""
