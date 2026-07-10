"""Stateful, meeting-local tools for the moderator in mode="moderator".

Unlike round_tools.py/research_agent.tools.roles (stateless handlers reading from the
per-run `runtime` dict, registered once into the shared global registry), these are
closures over one meeting's own mutable state (roster, notes, agenda, steps) -- each
call to build_moderator_tools() gets its own fresh ToolRegistry + state dict, since a
moderator meeting needs real cross-call mutable bookkeeping that isn't safely shared
across concurrent/unrelated meetings the way stateless tools are.

The moderator gets ONLY these meeting-management tools plus role_list/role_load (browse
the role library) -- not the full browser/files/terminal registry research_agent.tools.
meeting's moderator gets. Its job is orchestration; if it needs information, it calls on
a participant who has real tools.
"""
from __future__ import annotations

from typing import Any

from research_agent import roles as roles_api
from research_agent.tools import load_builtin_tools, registry as global_registry
from research_agent.tools.registry import ToolRegistry, json_result
from research_agent.tools.roles import register_role_tools

from .config import MeetingConfig, ParticipantConfig
from .trajectory import summarize_turn_actions


def build_moderator_tools(
    meeting_id: str,
    config: MeetingConfig,
    checkpoint,
) -> tuple[ToolRegistry, dict[str, Any]]:
    from .runner import _execute_turn  # deferred: avoids a circular import with runner.py

    state: dict[str, Any] = {
        "roster": {},        # name -> ParticipantConfig-shaped dict
        "prior_turns": {},   # name -> last turn dict (for "you've spoken before" context)
        "notes": "",
        "agenda": "",
        "steps": [],          # participant-turn steps, appended by meeting_call_on
        "conclusion": None,
        "call_counter": 0,
        "moderator_session_id": None,  # set by _run_moderator once the agent is constructed
    }

    def _save_checkpoint() -> None:
        # Every checkpoint write fully rebuilds runs/<id>.json from these fields (it
        # doesn't merge with the previous write), so moderator_session_id must be
        # threaded through here too -- otherwise a closure-triggered checkpoint (e.g.
        # meeting_set_agenda) would silently wipe out the session_id _run_moderator set
        # earlier, breaking resume for any crash after that point.
        checkpoint(
            state["steps"], notes=state["notes"], agenda=state["agenda"],
            roster=state["roster"], moderator_session_id=state["moderator_session_id"],
        )

    def _add_participant(args: dict, runtime: dict) -> str:
        name = str(args.get("name") or "").strip()
        if not name:
            return json_result(success=False, error="name is required")
        if name in state["roster"]:
            return json_result(success=False, error=f"Participant '{name}' already exists")

        role_ref = args.get("role_ref")
        if role_ref:
            try:
                roles_api.load_role(role_ref)  # validate it exists before accepting
            except FileNotFoundError as exc:
                return json_result(success=False, error=str(exc))
            entry = {
                "name": name, "role_ref": role_ref, "role": "", "skills": "",
                "system_prompt": None, "model": None, "provider": None, "max_iterations": 8,
            }
        else:
            entry = {
                "name": name,
                "role_ref": None,
                "role": str(args.get("role") or ""),
                "skills": str(args.get("skills") or ""),
                "system_prompt": args.get("system_prompt"),
                "model": args.get("model"),
                "provider": args.get("provider"),
                "max_iterations": int(args.get("max_iterations") or 8),
            }
        state["roster"][name] = entry
        _save_checkpoint()
        return json_result(success=True, name=name, roster=list(state["roster"].keys()))

    def _call_on(args: dict, runtime: dict) -> str:
        name = str(args.get("participant") or "").strip()
        question = str(args.get("question") or "").strip()
        if name not in state["roster"]:
            return json_result(
                success=False,
                error=f"Unknown participant: {name!r}. Call meeting_add_participant first.",
                roster=list(state["roster"].keys()),
            )
        if not question:
            return json_result(success=False, error="question is required")

        participant = ParticipantConfig(**state["roster"][name])
        prior_turn = state["prior_turns"].get(name)

        parts: list[str] = []
        if state["agenda"]:
            parts.append(f"=== Meeting agenda ===\n{state['agenda']}")
        if state["notes"]:
            parts.append(f"=== Moderator notes ===\n{state['notes']}")
        if prior_turn:
            parts.append(
                f"=== Your own previous turn in this meeting: actions taken ===\n"
                f"{summarize_turn_actions(prior_turn)}\n\n"
                f"=== Your own previous turn: your answer ===\n{prior_turn.get('output') or ''}"
            )
        parts.append(f"=== The moderator is asking you ===\n{question}")
        user_message = "\n\n".join(parts)

        state["call_counter"] += 1
        turn = _execute_turn(
            participant, user_message, meeting_id, state["call_counter"], config.verbose,
            decided_by="moderator", round_aware=False,
        )

        state["steps"].append({
            "step_index": None,
            "step_start": turn["start_time"],
            "step_end": turn["end_time"],
            "decided_by": "moderator",
            "trigger_reason": f"moderator called on {name}",
            "turns": [turn],
        })
        state["prior_turns"][name] = turn
        _save_checkpoint()
        return json_result(success=True, participant=name, answer=turn["output"])

    def _set_agenda(args: dict, runtime: dict) -> str:
        state["agenda"] = str(args.get("agenda") or "").strip()
        _save_checkpoint()
        return json_result(success=True)

    def _add_notes(args: dict, runtime: dict) -> str:
        content = str(args.get("content") or "").strip()
        state["notes"] = (state["notes"] + f"\n\n{content}").strip() if state["notes"] else content
        _save_checkpoint()
        return json_result(success=True)

    def _conclude(args: dict, runtime: dict) -> str:
        conclusion = str(args.get("conclusion") or "").strip()
        state["conclusion"] = conclusion
        runtime["final_response"] = conclusion
        _save_checkpoint()
        return json_result(success=True, conclusion=conclusion)

    load_builtin_tools()
    register_role_tools()
    # Keep ONLY role_list/role_load from the full shared registry -- exclude everything
    # else (browser/files/terminal/role_create/role_memory/kanban/...).
    keep = {"role_list", "role_load"}
    registry = global_registry.without(global_registry.names - keep)

    registry.register(
        "meeting_add_participant",
        {
            "description": (
                "Add a participant to the meeting. Either pull in an existing role via "
                "role_ref (see role_list/role_load), or define one ad-hoc with the other fields."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role_ref": {"type": "string", "description": "Name of a stored role to pull in."},
                    "role": {"type": "string", "description": "Ad-hoc: short role/perspective description."},
                    "skills": {"type": "string", "description": "Ad-hoc: skills/knowledge to include."},
                    "system_prompt": {"type": "string", "description": "Ad-hoc: full system prompt override."},
                    "model": {"type": "string"},
                    "provider": {"type": "string"},
                    "max_iterations": {"type": "integer"},
                },
                "required": ["name"],
            },
        },
        _add_participant,
    )
    registry.register(
        "meeting_call_on",
        {
            "description": (
                "Ask one participant one question and get their answer. You decide who "
                "speaks and in what order. The participant must already be in the roster "
                "(meeting_add_participant first)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "participant": {"type": "string"},
                    "question": {"type": "string"},
                },
                "required": ["participant", "question"],
            },
        },
        _call_on,
    )
    registry.register(
        "meeting_set_agenda",
        {
            "description": "Set the meeting agenda. Included in every participant's context.",
            "parameters": {"type": "object", "properties": {"agenda": {"type": "string"}}, "required": ["agenda"]},
        },
        _set_agenda,
    )
    registry.register(
        "meeting_add_notes",
        {
            "description": (
                "Append to your rolling meeting notes. Included in every subsequent "
                "participant's context -- keep this updated with whatever the next "
                "speaker needs to know (you do not have access to raw transcripts, only "
                "these notes plus each participant's own prior turn)."
            ),
            "parameters": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]},
        },
        _add_notes,
    )
    registry.register(
        "meeting_conclude",
        {
            "description": (
                "End the meeting with a synthesized final answer. This IS the final "
                "response -- there is no separate aggregation step in this mode."
            ),
            "parameters": {"type": "object", "properties": {"conclusion": {"type": "string"}}, "required": ["conclusion"]},
        },
        _conclude,
    )

    return registry, state
