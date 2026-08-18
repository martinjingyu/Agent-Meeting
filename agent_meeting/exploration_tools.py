"""ask_user_question: lets the pre-roster exploration agent (exploration.py) stop and
ask a real human a concrete, multiple-choice-plus-"Other" question, in the same
tool-call-accumulates-then-finish shape as role_architect_tools.submit_domain_brief --
the agent can call this any number of times, interleaved with its own reasoning,
before finishing the ordinary way (respond_to_user).

The handler blocks on real terminal input (agent_meeting.interactive.ask_choice) --
this only ever runs in the synchronous, human-attended exploration phase before a
meeting starts, never inside a participant's turn during the meeting itself.
"""
from __future__ import annotations

import contextlib

from research_agent.tools.registry import json_result, registry
from research_agent.tui import LiveDashboard

from .interactive import ask_choice


def _ask_user_question(args: dict, runtime: dict) -> str:
    question = str(args.get("question") or "").strip()
    options = [str(o).strip() for o in (args.get("options") or []) if str(o).strip()]

    if not question:
        return json_result(success=False, error="question must not be empty")
    if len(options) < 2:
        return json_result(
            success=False,
            error="options must contain at least 2 concrete candidate answers -- "
            "an 'Other' free-text slot is added automatically, do not include it yourself",
        )

    # The exploration agent runs through the same ConsoleUI/LiveDashboard as any
    # other GeneralAgent turn, which redraws a live status panel every 0.25s via raw
    # cursor-movement ANSI codes -- left running, it races with this blocking
    # input() prompt and corrupts both. LiveDashboard.active() returns None (a
    # no-op contextmanager below) rather than starting one when nothing is running.
    dashboard = LiveDashboard.active()
    pause_cm = dashboard.paused() if dashboard is not None else contextlib.nullcontext()
    with pause_cm:
        answer = ask_choice(
            question,
            options,
            header="=== Exploration phase: clarifying question before the roster is designed ===",
        )

    qa_log: list[dict] = runtime.setdefault("clarification_qa", [])
    qa_log.append({"question": question, "options": options, "answer": answer})

    return json_result(
        success=True,
        answer=answer,
        questions_asked_so_far=len(qa_log),
        status=(
            "Answer recorded. Ask another clarifying question if a real ambiguity "
            "remains, otherwise call respond_to_user with a summary of what was "
            "confirmed."
        ),
    )


def register_exploration_tools() -> None:
    registry.register(
        "ask_user_question",
        {
            "description": (
                "Ask the human stakeholder one concrete clarifying question before the "
                "meeting's participant roster is designed, with 2-4 concrete candidate "
                "answers (a free-text 'Other' option is offered automatically, do not "
                "list it yourself). Call this once per question, interleaved with your "
                "own reasoning -- do not batch every question into one call. Only ask "
                "about a genuine ambiguity in the task spec that would change what the "
                "roster or the meeting should focus on; do not ask about something the "
                "task spec already answers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "A single, specific, human-understandable question.",
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "2-4 concrete candidate answers, written so a non-expert "
                            "stakeholder can pick one without needing to write anything."
                        ),
                    },
                },
                "required": ["question", "options"],
            },
        },
        _ask_user_question,
    )
