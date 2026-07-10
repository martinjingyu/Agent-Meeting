"""Smoke test: three roles answer one question in parallel, then get synthesized.

Run with:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\run_parallel_qa.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_meeting import MeetingConfig, ParticipantConfig, run_meeting


def main() -> None:
    config = MeetingConfig(
        question="Should a small SaaS startup build its own analytics pipeline or buy one?",
        participants=[
            ParticipantConfig(name="Engineer", role="Pragmatic backend engineer"),
            ParticipantConfig(name="Finance", role="Cost-conscious finance lead"),
            ParticipantConfig(name="Product", role="Product manager focused on time-to-market"),
        ],
        aggregation_strategy="llm",
    )
    result = run_meeting(config)
    print(f"meeting_id: {result['meeting_id']}")
    print(f"saved to: runs/{result['meeting_id']}.json")
    print("\n=== final_response ===\n")
    print(result["final_response"])


if __name__ == "__main__":
    main()
