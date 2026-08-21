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

ask_user_questions takes a whole batch of questions in one call, so the human sees
and answers everything the judge currently has for them in one sitting instead of a
slow one-question, one-tool-call-round-trip-at-a-time drip. It is still a genuine
back-and-forth across calls, not one batch-then-done: the judge can call it again
after reading a batch's answers if those answers raise a genuine follow-up, and if
any single answer is itself a question or shows confusion, the tool's own result
tells the judge to clarify and re-ask that one in a later call rather than record
confusion as a real answer. Same TUI and same LiveDashboard-pausing concern as
exploration_tools.py's ask_user_question, same accumulate-into-runtime pattern.

submit_judgment is the finish tool -- same accumulate-then-finish shape as
role_architect_tools.submit_domain_brief / round_tools.submit_round_answer, ending
the turn via runtime["final_response"] (see research_agent/agent.py's loop-end
check) the same way respond_to_user does.
"""
from __future__ import annotations

import contextlib

from research_agent.tools.registry import ToolRegistry, json_result
from research_agent.tui import LiveDashboard

from .interactive import ask_choices


def _ask_user_questions(args: dict, runtime: dict) -> str:
    raw_questions = args.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        return json_result(success=False, error="questions must be a non-empty array")

    parsed: list[tuple[str, list[str]]] = []
    for item in raw_questions:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question") or "").strip()
        if not q:
            continue
        opts = [str(o).strip() for o in (item.get("options") or []) if str(o).strip()]
        parsed.append((q, opts))
    if not parsed:
        return json_result(success=False, error="no valid questions found in questions array")

    # A meeting's live progress display(s) redraw on a timer via raw terminal
    # control codes and would race with these blocking input() prompts if left
    # running -- see interactive.py's module docstring and exploration_tools.py's
    # identical guard. runtime["judge_meeting_progress"] is threaded in by
    # judge.py's run_interactive_judge (extra_runtime=); LiveDashboard is a
    # process-wide singleton, checked directly. Paused once for the whole batch,
    # not per-question -- the human answers every question in this call in one
    # sitting, so there's no reason to let the display resume and re-pause between
    # them.
    progress = runtime.get("judge_meeting_progress")
    dashboard = LiveDashboard.active()
    progress_cm = progress.paused() if progress is not None else contextlib.nullcontext()
    dashboard_cm = dashboard.paused() if dashboard is not None else contextlib.nullcontext()
    qa_log: list[dict] = runtime.setdefault("judge_qa", [])
    with progress_cm, dashboard_cm:
        # ask_choices prints every question in the batch up front (so the human sees
        # everything before answering any of it), then collects each answer in turn --
        # unlike looping ask_choice per question, which would only ever show one
        # question at a time.
        raw_answers = ask_choices(parsed, header="=== The judge has questions for you ===")
    answered: list[dict] = []
    for (question, options), answer in zip(parsed, raw_answers):
        qa_log.append({"question": question, "options": options, "answer": answer})
        answered.append({"question": question, "answer": answer})

    return json_result(
        success=True,
        answers=answered,
        status=(
            "All answers recorded, in order. If any answer is itself a question, "
            "asks you to explain or rephrase, or otherwise shows they didn't "
            "understand what you asked -- do NOT treat it as a real answer for that "
            "one. Instead, in your next turn, answer their question / clarify what "
            "you meant, then call ask_user_questions again to re-ask just that item "
            "(bundled with any other genuine follow-up the rest of the batch raised) "
            "with a clearer version. Keep doing this for as many rounds as it takes. "
            "Only once every question has a real, usable answer should you call "
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
        "ask_user_questions",
        {
            "description": (
                "Ask the human stakeholder a whole batch of concrete questions in one "
                "call, each with 2-4 concrete candidate answers (a free-text 'Other' "
                "option is offered automatically per question, do not list it "
                "yourself). They answer every question in the batch in one sitting, "
                "in order, and you get all the answers back together. Ask about "
                "everything genuinely material this round raised in your first call -- "
                "do not artificially limit yourself to one question or space "
                "questions out one per call. You may still call this again after "
                "reading a batch's answers if they raise a genuine follow-up, "
                "including re-asking (in a clearer form) any single item whose answer "
                "showed the human didn't understand it -- keep going for as many "
                "calls as it takes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "description": "One or more questions to ask in this batch.",
                        "items": {
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
                                        "2-4 concrete candidate answers, written so a "
                                        "non-expert stakeholder can pick one without "
                                        "needing to write anything."
                                    ),
                                },
                            },
                            "required": ["question", "options"],
                        },
                    },
                },
                "required": ["questions"],
            },
        },
        _ask_user_questions,
    )
    registry.register(
        "submit_judgment",
        {
            "description": (
                "Finish your review of this round with a final stop/continue verdict. "
                "You MUST call this to end your turn -- there is no other way to "
                "finish. Call it only after ask_user_questions has already asked and "
                "gotten an answer to the mandatory final stop/continue confirmation "
                "question (see system prompt) -- there is no zero-question path to "
                "this tool."
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
