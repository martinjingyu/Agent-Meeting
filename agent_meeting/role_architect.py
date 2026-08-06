"""Designs the roster of technical-domain participants for a planning_rounds meeting
from the task spec + Stage-1 evidence, instead of a human hand-picking the domains
up front (see examples/plan_image_gallery_technical_domains_auto_roster.py). This is
the missing piece that makes that example's pipeline end-to-end: task + evidence in,
a ready-to-run participant roster out.

One continuous GeneralAgent tool loop does the whole job -- deciding how many
domains this specific task needs, what each one is, and writing its full brief --
rather than splitting "decide the roster" and "write each brief" into separate LLM
calls. It has file-tool access to the Stage-1 evidence reports (evidence_paths) and
decides for itself what's worth opening, and submits each domain via
submit_domain_brief (role_architect_tools.py) as it finalizes it -- so it can revise
an earlier domain's boundary the moment writing a later one reveals an overlap,
instead of a fixed decomposition made before any brief existed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_agent.agent import GeneralAgent

from .role_architect_tools import register_role_architect_tools
from .storage import participant_workspace_dir
from .tools_setup import build_participant_registry
from .trajectory import log

ROLE_ARCHITECT_MODEL = "gpt-5.5"
ROLE_ARCHITECT_PROVIDER = "codex"
ROLE_ARCHITECT_REASONING_EFFORT = "high"
ROLE_ARCHITECT_MAX_ITERATIONS = 40


class RosterDesignError(Exception):
    """The architect finished without a usable roster (out of the requested size
    range, or nothing submitted at all). Raised rather than silently falling back to
    something wrong -- a broken roster is not a safe default."""


@dataclass
class DomainRole:
    name: str
    """Short PascalCase-style identifier -- becomes the literal ParticipantConfig.name
    used everywhere downstream (transcript headers, shared subfolder name via
    storage.participant_shared_dir, etc.); enforced by submit_domain_brief's own
    validation, not re-checked here."""
    domain_brief: str
    """Paragraph describing what this domain covers, what to consider, and what not
    to assume -- written in the same style as this project's DOMAIN_BRIEFS entries,
    meant to be dropped straight into a participant's system prompt."""


def _build_prompt(
    task_spec: str,
    technology_reference: str | None,
    evidence_paths: list[Path],
    min_roles: int,
    max_roles: int,
) -> tuple[str, str]:
    reference_section = (
        f"\n\n=== Technology reference (a menu, not a required list or an ordering) ===\n"
        f"{technology_reference}"
        if technology_reference
        else ""
    )
    evidence_section = (
        "\n".join(f"- {p}" for p in evidence_paths)
        if evidence_paths
        else "(no evidence files given -- rely on the task spec and reference menu below)"
    )
    system_prompt = (
        "You are designing, from scratch, the roster of technical-domain "
        "participants for a multi-round planning meeting -- in one continuous "
        "working session, not a fixed template.\n\n"
        "Your job has two parts, interleaved as you see fit rather than done as two "
        "separate passes:\n"
        "1. Decide the domains: read the task spec, and use your file tools to "
        "inspect whichever Stage-1 evidence reports below are actually relevant "
        f"(not necessarily all of them) before deciding. Produce between {min_roles} "
        f"and {max_roles} domains -- use your own judgment on the right number for "
        "this task's real complexity, do not default to the midpoint. Domains must "
        "be mutually distinct (no two should own materially the same technical "
        "decision), collectively covering (between them, span every technical "
        "decision that materially affects the final solution -- do not leave an "
        "important one, including a one-off setup duty like a shared environment "
        "survey, with no owner), and argued from a real technical stance rather "
        "than a project-management role.\n"
        "2. Write each domain's full brief and call submit_domain_brief(name, "
        "domain_brief) as you finalize it -- do this per domain as you go, not all "
        "at the end. The brief is a paragraph written directly to the participant "
        "who will receive it as their own system prompt: their domain's scope, the "
        "concrete techniques/considerations relevant to it, and what assumption or "
        "shortcut to specifically avoid. If writing a later domain reveals an "
        "earlier one's boundary was wrong, call submit_domain_brief again with that "
        "earlier name to revise it -- you are not locked into your first pass.\n\n"
        "Writing style for the brief: write for a human reader, not a spec sheet. "
        "Explain each technique in a real clause (what it's for, why it matters "
        "here), not as a bare comma/slash-separated term dump -- 'perceptual hashes "
        "like pHash to catch resized or recompressed copies' reads far better than "
        "'pHash/dHash/colorHash/SSIM'. Prefer several shorter, plain sentences over "
        "one long chain of semicolon-joined clauses. Every technical term you use "
        "should be doing real work for the reader, not padding for thoroughness -- a "
        "reader who is competent but not already expert in this exact subfield "
        "should be able to follow the whole paragraph on one read.\n\n"
        f"Available Stage-1 evidence reports:\n{evidence_section}"
        f"{reference_section}\n\n"
        "Only call respond_to_user, with a short final summary of the roster you "
        "designed, once the tool result confirms every domain has been submitted "
        "and you're satisfied they collectively cover the task."
    )
    user_message = f"=== Task spec ===\n{task_spec}\n\nDesign the roster now."
    return system_prompt, user_message


def design_domain_roster(
    task_spec: str,
    meeting_id: str,
    *,
    technology_reference: str | None = None,
    evidence_paths: list[Path] | None = None,
    min_roles: int = 4,
    max_roles: int = 8,
    model: str = ROLE_ARCHITECT_MODEL,
    provider: str = ROLE_ARCHITECT_PROVIDER,
    reasoning_effort: str = ROLE_ARCHITECT_REASONING_EFFORT,
    max_iterations: int = ROLE_ARCHITECT_MAX_ITERATIONS,
    verbose: bool = True,
) -> list[DomainRole]:
    """meeting_id is required even though this runs before the meeting proper: the
    agent gets a real workspace_root, scoped via
    storage.participant_workspace_dir(meeting_id, "RoleArchitect"), consistent with
    how every other agent in a meeting gets its workspace. Raises RosterDesignError
    rather than returning a partial/best-effort roster -- callers should let this
    propagate and fail the run rather than catch it."""
    register_role_architect_tools()
    system_prompt, user_message = _build_prompt(
        task_spec, technology_reference, evidence_paths or [], min_roles, max_roles
    )

    if verbose:
        log("role-architect", "designing roster + writing briefs in one session...")

    registry = build_participant_registry(role_backed=False, round_aware=False)
    agent = GeneralAgent(
        model=model,
        provider=provider,
        reasoning_effort=reasoning_effort,
        max_iterations=max_iterations,
        self_review=False,
        registry=registry,
        sub_agent=True,
        agent_role="role_architect",
        workspace_root=participant_workspace_dir(meeting_id, "RoleArchitect"),
        extra_runtime={
            "role_architect_min_roles": min_roles,
            "role_architect_max_roles": max_roles,
        },
    )
    agent.run(user_message, system_prompt=system_prompt)
    briefs: dict[str, str] = getattr(agent, "_runtime", {}).get("domain_briefs") or {}

    if not (min_roles <= len(briefs) <= max_roles):
        raise RosterDesignError(
            f"role architect submitted {len(briefs)} domain(s) ({sorted(briefs)}), "
            f"expected between {min_roles} and {max_roles}"
        )
    if verbose:
        log("role-architect", f"designed {len(briefs)} domain(s): {', '.join(sorted(briefs))}")
    return [DomainRole(name=name, domain_brief=brief) for name, brief in briefs.items()]
