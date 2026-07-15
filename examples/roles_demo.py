"""Role management demo: create two reusable roles, run a meeting referencing them
by name, then run a second meeting to show each role's memory carrying over.

Run with:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\roles_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent_meeting import MeetingConfig, ParticipantConfig, roles as roles_api, run_meeting


def ensure_roles() -> None:
    if "skeptic-reviewer" not in roles_api.list_roles():
        roles_api.create_role(
            "skeptic-reviewer",
            description="Skeptical peer reviewer -- pushes back on vague claims",
            persona="A skeptical peer reviewer who has seen hundreds of half-baked proposals.",
            constraints=["Never accept a claim without asking what evidence would falsify it."],
            output_contract="End every review with a one-line verdict: ACCEPT | REVISE | REJECT.",
            style="Terse. No hedging language.",
        )
    if "risk-extractor" not in roles_api.list_roles():
        roles_api.create_role(
            "risk-extractor",
            description="Purely functional role -- no persona, just extracts risks",
            purpose="Extract concrete, falsifiable risk factors from whatever is being discussed.",
            output_contract="A bullet list of risks, each one sentence, no commentary.",
        )


def run_once(label: str) -> None:
    config = MeetingConfig(
        question="We're proposing to let meeting participants write to their own persistent "
        "memory via a role_memory tool. What's the strongest objection to this design?",
        participants=[
            ParticipantConfig(name="Reviewer", role_ref="skeptic-reviewer", max_iterations=6),
            ParticipantConfig(name="RiskExtractor", role_ref="risk-extractor", max_iterations=6),
        ],
        aggregation_strategy="concat",
    )
    result = run_meeting(config)
    print(f"\n=== {label} (meeting_id={result['meeting_id']}) ===")
    for turn in result["steps"][0]["turns"]:
        print(f"\n--- {turn['agent']} ({turn['role_ref']}) ---\n{turn['output']}")


def main() -> None:
    ensure_roles()
    run_once("first run")
    print("\n\nskeptic-reviewer memory after first run:", roles_api.role_memory_entries(roles_api.load_role("skeptic-reviewer")))
    run_once("second run (should see prior memory in its input if the role wrote any)")


if __name__ == "__main__":
    main()
