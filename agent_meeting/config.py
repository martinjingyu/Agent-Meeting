from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParticipantConfig:
    name: str
    role: str = ""
    skills: str = ""
    system_prompt: str | None = None
    model: str | None = None
    provider: str | None = None
    max_iterations: int = 8
    role_ref: str | None = None
    """Name of a stored role (agent_meeting.roles) to pull in instead of the ad-hoc
    fields above. When set, name/role/skills/system_prompt are ignored -- the role's
    own DEFINITION.md + persistent memory drive the participant's identity, and
    model/provider/max_iterations fall back to the role's own frontmatter defaults
    when left unset here."""

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
