"""Tools for the interactive judge (judge.py's run_interactive_judge, used when
MeetingConfig.human_checkin is True): exactly two tools, matching the judge's two
jobs and nothing else -- it is not a general-purpose agent, it should not gain file
access, search, or any other capability beyond talking to the human and recording a
verdict. build_judge_registry() builds a fresh, isolated ToolRegistry containing only
these two -- deliberately NOT registered into research_agent's shared global registry
singleton (unlike every other *_tools.py in this project), and NOT built from
build_participant_registry()/load_builtin_tools(), since either of those would hand
the judge every builtin tool (files, terminal, search, ...) filtered down after the
fact rather than actually restricting it to two.

ask_user_question is a genuine back-and-forth, not one question-then-done: the judge
can call it as many times as it needs, and if the human's "answer" is itself a
question or shows confusion, the tool's own result tells the judge to clarify and
ask again rather than record confusion as a real answer. Same TUI and same
LiveDashboard-pausing concern as exploration_tools.py's ask_user_question, same
accumulate-into-runtime pattern.

submit_judgment is the finish tool -- same accumulate-then-finish shape as
role_architect_tools.submit_domain_brief / round_tools.submit_round_answer, ending
the turn via runtime["final_response"] (see research_agent/agent.py's loop-end
check) the same way respond_to_user does.
"""
from __future__ import annotations

import contextlib

from research_agent.tools.registry import ToolRegistry, json_result
from research_agent.tui import LiveDashboard

from .interactive import ask_choice


def _ask_user_question(args: dict, runtime: dict) -> str:
    question = str(args.get("question") or "").strip()
    options = [str(o).strip() for o in (args.get("options") or []) if str(o).strip()]

    if not question:
        return json_result(success=False, error="question must not be empty")

    # A meeting's live progress display(s) redraw on a timer via raw terminal
    # control codes and would race with this blocking input() prompt if left
    # running -- see interactive.py's module docstring and exploration_tools.py's
    # identical guard. runtime["judge_meeting_progress"] is threaded in by
    # judge.py's run_interactive_judge (extra_runtime=); LiveDashboard is a
    # process-wide singleton, checked directly.
    progress = runtime.get("judge_meeting_progress")
    dashboard = LiveDashboard.active()
    progress_cm = progress.paused() if progress is not None else contextlib.nullcontext()
    dashboard_cm = dashboard.paused() if dashboard is not None else contextlib.nullcontext()
    with progress_cm, dashboard_cm:
        answer = ask_choice(question, options, header="=== The judge has a question for you ===")

    qa_log: list[dict] = runtime.setdefault("judge_qa", [])
    qa_log.append({"question": question, "options": options, "answer": answer})

    return json_result(
        success=True,
        answer=answer,
        status=(
            "Answer recorded. If that answer is itself a question, asks you to "
            "explain or rephrase, or otherwise shows they didn't understand what you "
            "asked -- do NOT treat it as a real answer. Instead, answer their "
            "question / clarify what you meant, then call ask_user_question again "
            "with a clearer version of your original question. Keep doing this for "
            "as many exchanges as it takes. Only once you have a real, usable answer "
            "(or you decide you don't need to ask anything at all) should you call "
            "submit_judgment to finish."
        ),
    )


def _submit_judgment(args: dict, runtime: dict) -> str:
    stop = bool(args.get("stop"))
    reasoning = str(args.get("reasoning") or "").strip()
    unresolved_issues = [str(x).strip() for x in (args.get("unresolved_issues") or []) if str(x).strip()]
    per_participant_coverage = {
        str(k): str(v) for k, v in (args.get("per_participant_coverage") or {}).items()
    }

    if not reasoning:
        return json_result(success=False, error="reasoning must not be empty")

    runtime["judgment"] = {
        "stop": stop,
        "reasoning": reasoning,
        "unresolved_issues": unresolved_issues,
        "per_participant_coverage": per_participant_coverage,
    }
    runtime["final_response"] = reasoning
    return json_result(success=True, message="Judgment recorded, turn finished.")


def build_judge_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        "ask_user_question",
        {
            "description": (
                "Ask the human stakeholder one concrete question, with 2-4 concrete "
                "candidate answers (a free-text 'Other' option is offered "
                "automatically, do not list it yourself). You may call this as many "
                "times as you need -- including follow-ups if their answer shows they "
                "didn't understand your question, in which case clarify and ask "
                "again rather than guessing at what they meant."
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
    registry.register(
        "submit_judgment",
        {
            "description": (
                "Finish your review of this round with a final stop/continue verdict. "
                "You MUST call this to end your turn -- there is no other way to "
                "finish. Call it only once you're done asking the human anything you "
                "needed to (zero questions is fine if none were needed)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "per_participant_coverage": {
                        "type": "object",
                        "description": (
                            "Maps each participant name to one sentence on whether their "
                            "contribution this round was substantive or a placeholder/no-op."
                        ),
                    },
                    "unresolved_issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Blocking discussion gaps that another participant round could "
                            "materially improve; do not list mere Planner synthesis choices "
                            "here. Empty list if genuinely none."
                        ),
                    },
                    "reasoning": {
                        "type": "string",
                        "description": (
                            "A few sentences explaining the stop/continue call, referencing "
                            "unresolved_issues and any human answer you received, and "
                            "explicitly separating discussion gaps from Planner synthesis "
                            "decisions."
                        ),
                    },
                    "stop": {
                        "type": "boolean",
                        "description": "true if the meeting should stop now, false to continue.",
                    },
                },
                "required": ["reasoning", "stop"],
            },
        },
        _submit_judgment,
    )
    return registry
