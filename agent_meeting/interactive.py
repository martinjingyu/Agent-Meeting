"""A small terminal question TUI in the style of Claude Code's AskUserQuestion tool:
a numbered list of concrete candidate answers plus an always-available "Other" slot
for free text. Used in two places:

- exploration.py's pre-roster exploration agent calls this (via a registered tool,
  see exploration_tools.py) to interactively confirm scope/ambiguities with a human
  before any participant role is designed. One question at a time (ask_choice).
- judge_tools.py's ask_user_questions tool, for MeetingConfig.human_checkin's
  interactive judge, batches several questions into one call (ask_choices): the whole
  batch is shown up front so the human can see everything being asked before
  answering any of it, then each is answered in turn.

Deliberately just `input()`/`print()` -- no new dependency (rich is already a
dependency via tui.py, but this needs to actually block for real terminal input,
which a Live-rendered dashboard is not built for), and both call sites already only
run when a human is known to be at the keyboard.
"""
from __future__ import annotations

_OTHER_LABEL = "Other (type your own answer)"


def _prompt_choice(prompt: str, options: list[str]) -> str:
    """Shared input-validation loop for one already-displayed question: reads a
    number and returns the corresponding option's exact text (prompting again for
    free text if the "Other" slot -- the last entry in `options` -- is chosen), or
    loops back on invalid/empty input. `options` here already includes the trailing
    "Other" label; callers display it themselves (ask_choice inline, ask_choices as
    part of the batch preview) since where/how it's numbered differs between the
    two."""
    while True:
        raw = input(prompt).strip()
        if not raw.isdigit():
            print(f"Please enter a number between 1 and {len(options)}.")
            continue
        idx = int(raw)
        if not (1 <= idx <= len(options)):
            print(f"Please enter a number between 1 and {len(options)}.")
            continue
        if idx == len(options):
            free = input("Your answer: ").strip()
            if not free:
                print("Please type a non-empty answer.")
                continue
            return free
        return options[idx - 1]


def ask_choice(
    question: str, options: list[str], *, header: str | None = None, progress: str | None = None,
) -> str:
    """Render `question` with `options` as a numbered menu plus a trailing "Other"
    choice, block for input, and return the chosen option's exact text -- or the
    user's own free-text answer if they pick "Other". Falls back to a bare free-text
    prompt if `options` is empty.

    `progress` (e.g. "Question 2 of 5"), when given, prints above `header`."""
    print("\n" + "=" * 72)
    if progress:
        print(progress)
    if header:
        print(header)
    print(f"? {question}")
    print("=" * 72)

    if not options:
        answer = input("\nYour answer: ").strip()
        while not answer:
            answer = input("Please type a non-empty answer: ").strip()
        return answer

    display_options = [*options, _OTHER_LABEL]
    for i, opt in enumerate(display_options, 1):
        print(f"  {i}. {opt}")
    return _prompt_choice(f"\nYour choice [1-{len(display_options)}]: ", display_options)


def ask_choices(questions: list[tuple[str, list[str]]], *, header: str | None = None) -> list[str]:
    """Batch version of ask_choice: shows every question and its numbered options up
    front in one block -- so the human sees the whole batch before answering
    anything, not just the one currently in front of them -- then prompts for each
    answer in turn, referencing back to the question already printed above rather
    than re-printing its full option list. Questions with no options fall back to a
    bare free-text prompt, same as ask_choice. Returns answers in the same order as
    `questions`."""
    print("\n" + "=" * 72)
    if header:
        print(header)
    print(f"{len(questions)} question(s):")
    print("=" * 72)
    display_options_per_q: list[list[str] | None] = []
    for i, (question, options) in enumerate(questions, 1):
        print(f"\nQ{i}. {question}")
        if not options:
            display_options_per_q.append(None)
            continue
        display_options = [*options, _OTHER_LABEL]
        display_options_per_q.append(display_options)
        for j, opt in enumerate(display_options, 1):
            print(f"  {j}. {opt}")
    print("\n" + "=" * 72)

    answers: list[str] = []
    for i, (question, _options) in enumerate(questions, 1):
        display_options = display_options_per_q[i - 1]
        short = question if len(question) <= 60 else question[:59] + "…"
        if display_options is None:
            answer = input(f"\nYour answer to Q{i} ({short}): ").strip()
            while not answer:
                answer = input("Please type a non-empty answer: ").strip()
            answers.append(answer)
            continue
        answer = _prompt_choice(
            f"\nYour choice for Q{i} ({short}) [1-{len(display_options)}]: ", display_options,
        )
        answers.append(answer)
    return answers
