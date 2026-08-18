"""A small terminal question TUI in the style of Claude Code's AskUserQuestion tool:
a numbered list of concrete candidate answers plus an always-available "Other" slot
for free text. Used in two places:

- exploration.py's pre-roster exploration agent calls this (via a registered tool,
  see exploration_tools.py) to interactively confirm scope/ambiguities with a human
  before any participant role is designed.
- runner.py's planning_rounds loop calls this directly (not through a tool -- the
  judge is a single LLM call, not a GeneralAgent) when MeetingConfig.human_checkin is
  True and a round's judge produced a question_for_user.

Deliberately just `input()`/`print()` -- no new dependency (rich is already a
dependency via tui.py, but this needs to actually block for real terminal input,
which a Live-rendered dashboard is not built for), and both call sites already only
run when a human is known to be at the keyboard.
"""
from __future__ import annotations

_OTHER_LABEL = "Other (type your own answer)"


def ask_choice(question: str, options: list[str], *, header: str | None = None) -> str:
    """Render `question` with `options` as a numbered menu plus a trailing "Other"
    choice, block for input, and return the chosen option's exact text -- or the
    user's own free-text answer if they pick "Other". Falls back to a bare free-text
    prompt if `options` is empty."""
    print("\n" + "=" * 72)
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

    while True:
        raw = input(f"\nYour choice [1-{len(display_options)}]: ").strip()
        if not raw.isdigit():
            print(f"Please enter a number between 1 and {len(display_options)}.")
            continue
        idx = int(raw)
        if not (1 <= idx <= len(display_options)):
            print(f"Please enter a number between 1 and {len(display_options)}.")
            continue
        if idx == len(display_options):
            free = input("Your answer: ").strip()
            if not free:
                print("Please type a non-empty answer.")
                continue
            return free
        return options[idx - 1]
