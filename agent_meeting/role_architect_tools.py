"""submit_domain_brief: lets the role-architect agent (role_architect.py) hand back
one finished domain -- name + full brief -- at a time, within a single ongoing tool
loop, without ending its turn each time. Unlike round_tools.py's submit_round_answer,
which is a *finish* tool (setting runtime["final_response"] ends the turn the moment
it's called), this one just accumulates into runtime["domain_briefs"] and lets the
agent keep deciding/writing domains until it's satisfied the roster is complete, then
it finishes the ordinary way (respond_to_user).

Registered into the same global research_agent.tools.registry singleton that
build_participant_registry() filters from (mirrors round_tools.py's opt-in pattern),
kept local to Agent-Meeting since it's specific to this one workflow.
"""
from __future__ import annotations

import re

from research_agent.tools.registry import json_result, registry

_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,29}$")


def _submit_domain_brief(args: dict, runtime: dict) -> str:
    name = str(args.get("name") or "").strip()
    domain_brief = str(args.get("domain_brief") or "").strip()

    if not _NAME_RE.match(name):
        return json_result(
            success=False,
            error=(
                f"{name!r} is not a valid domain name -- must be a short PascalCase "
                "identifier (letters/digits/underscore only, <=30 chars)"
            ),
        )
    if not domain_brief:
        return json_result(success=False, error="domain_brief must not be empty")

    briefs: dict[str, str] = runtime.setdefault("domain_briefs", {})
    is_revision = name in briefs
    briefs[name] = domain_brief

    min_roles = runtime.get("role_architect_min_roles")
    max_roles = runtime.get("role_architect_max_roles")
    count = len(briefs)
    if max_roles is not None and count >= max_roles:
        status = f"{count}/{max_roles} domains submitted (at the max) -- stop here and finish."
    elif min_roles is not None and count < min_roles:
        status = f"{count} domain(s) submitted so far, at least {min_roles} needed before finishing."
    else:
        status = (
            f"{count} domain(s) submitted so far (between {min_roles} and {max_roles} is fine) -- "
            "add another if the roster still leaves a real technical decision uncovered, "
            "otherwise call respond_to_user to finish."
        )
    return json_result(success=True, revised=is_revision, submitted_count=count, status=status)


def register_role_architect_tools() -> None:
    registry.register(
        "submit_domain_brief",
        {
            "description": (
                "Submit one finished domain for this meeting's participant roster: its "
                "name and full brief. Call this once per domain as you finalize each "
                "one (not all at the end) -- you can revise a domain by calling this "
                "again with the same name. Only call respond_to_user, with a short "
                "final summary, once the tool result tells you the roster is complete."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Short PascalCase identifier for this domain (letters/digits/"
                            "underscore only, <=30 chars) -- becomes the participant's "
                            "literal display name in the meeting transcript."
                        ),
                    },
                    "domain_brief": {
                        "type": "string",
                        "description": (
                            "The full brief paragraph for this domain -- written directly to "
                            "the participant who will receive it as their own system prompt: "
                            "its scope, the concrete techniques/considerations relevant to it, "
                            "and what assumption or shortcut it should specifically avoid."
                        ),
                    },
                },
                "required": ["name", "domain_brief"],
            },
        },
        _submit_domain_brief,
    )
