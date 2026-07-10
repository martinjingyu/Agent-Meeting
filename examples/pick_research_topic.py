"""Meta example: use one round of parallel_qa + LLM aggregation to pick a good
agent/LLM-related research topic that this very framework (Agent-Meeting) is well
suited to investigate.

Run with:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\pick_research_topic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent_meeting import MeetingConfig, ParticipantConfig, run_meeting

QUESTION = """\
We just built Agent-Meeting: a backend for running multi-agent "meeting" experiments.
Today it supports one mode — several role-agents (full GeneralAgent instances, with
real tools: browser, files, terminal, memory, etc.) answer the same question in
parallel, then an LLM aggregator synthesizes their answers into one final response.
Every step is logged with per-turn and per-tool-call timestamps, structured as
step -> turns -> events, with a `decided_by` field (script/moderator/config) so future
modes (an LLM moderator deciding who speaks when, or a hybrid of scripted + moderator
steps) can be added without changing the log schema. A timeline visualization frontend
is planned but not built yet.

Propose ONE concrete, interesting agent/LLM research topic that this framework is
well suited to investigate. For your proposal, be specific about:
1. The research question and a falsifiable hypothesis.
2. Why parallel multi-agent discussion + aggregation (as opposed to a single agent,
   or a sequential/moderator-driven meeting) is the right setup to study it.
3. What you'd vary across runs (number of participants, role diversity, aggregation
   strategy, tool access, etc.) and what you'd measure.
4. What a minimal first experiment would look like using ONLY what's built today
   (i.e. do not assume the moderator mode or the frontend already exist).
"""


def main() -> None:
    config = MeetingConfig(
        question=QUESTION,
        participants=[
            ParticipantConfig(
                name="MultiAgentResearcher",
                role="Multi-agent systems researcher, deep familiarity with debate/"
                "society-of-mind/self-consistency literature",
            ),
            ParticipantConfig(
                name="LLMEvalResearcher",
                role="LLM evaluation methodologist, focused on benchmark design and "
                "avoiding confounds/leakage in experiments",
            ),
            ParticipantConfig(
                name="SystemsEngineer",
                role="Systems engineer familiar with this exact codebase, focused on "
                "what's actually feasible to measure with today's implementation",
            ),
            ParticipantConfig(
                name="SkepticReviewer",
                role="Skeptical peer reviewer who pushes back on vague hypotheses, "
                "unfalsifiable claims, and topics that aren't actually novel",
            ),
        ],
        aggregation_strategy="llm",
    )
    result = run_meeting(config)
    print(f"meeting_id: {result['meeting_id']}")
    print(f"saved to: runs/{result['meeting_id']}.json")
    for turn in result["steps"][0]["turns"]:
        print(f"\n=== {turn['agent']} ===\n{turn['output']}")
    print("\n=== final_response (aggregated) ===\n")
    print(result["final_response"])


if __name__ == "__main__":
    main()
