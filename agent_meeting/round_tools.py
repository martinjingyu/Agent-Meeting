"""Round-finish tool for round >= 2 of a multi-round meeting.

Round 1 participants finish via the normal research_agent `respond_to_user` tool.
From round 2 onward, participants finish via `submit_round_answer` instead -- its
schema makes `changes_from_prior_round` a REQUIRED argument, so the model cannot end
the turn without stating what changed (or explicitly "no changes"). This is a much
stronger constraint than asking for a summary in prose: function-call schema
validation, not a hope that the model remembers to include something.

Registered into the same global research_agent.tools.registry singleton that
build_participant_registry() filters from (mirrors agent_meeting.role_tools'
register_role_tools() opt-in pattern) -- kept local to Agent-Meeting rather than
added to the shared research_agent library, since it's specific to this project's
multi-round mechanic, not a general-purpose tool.
"""
from __future__ import annotations

from typing import Any

from research_agent.tools.registry import json_result, registry


def _finish_round(args: dict, runtime: dict) -> str:
    answer = str(args.get("answer") or "").strip()
    changes = str(args.get("changes_from_prior_round") or "").strip()
    runtime["final_response"] = answer
    runtime["round_changes"] = changes
    return json_result(success=True)


def register_round_tools() -> None:
    registry.register(
        "submit_round_answer",
        {
            "description": (
                "Submit your position for this round. You MUST call this to finish "
                "the round -- there is no other way to end your turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "changes_from_prior_round": {
                        "type": "string",
                        "description": (
                            "What you changed vs. your own previous round's position, and why. "
                            "Write 'No changes' if you are keeping your prior position as-is."
                        ),
                    },
                    "answer": {
                        "type": "string",
                        "description": "Your full answer/position for this round.",
                    },
                },
                "required": ["changes_from_prior_round", "answer"],
            },
        },
        _finish_round,
    )
