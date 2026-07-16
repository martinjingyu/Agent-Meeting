"""Judge for mode="planning_rounds": after each ideation round, decides whether the
discussion has surfaced enough for the Planner to synthesize a Plan, or whether
another round would meaningfully add something.

A single deterministic LLM call (same pattern as aggregate.py's _aggregate_llm),
not a GeneralAgent turn -- the judge only ever reads a transcript and answers a
strict 0/1, it never needs tools. Always DeepSeek v4 Pro at medium thinking,
matching the participants' default -- this is an internal gatekeeping decision, not
something a caller tunes per meeting the way aggregation_model/provider are.
"""
from __future__ import annotations

from research_agent.llm import LLMClient

JUDGE_MODEL = "deepseek-v4-pro"
JUDGE_PROVIDER = "deepseek"
JUDGE_REASONING_EFFORT = "medium"


def judge_should_stop(question: str, transcript: str) -> dict[str, str | bool]:
    prompt = (
        "You are the gatekeeper for a planning discussion. Below is the task and every "
        "round of participant contributions so far (points, suggestions, and ideas only "
        "-- no Plan has been written yet).\n\n"
        f"=== Task ===\n{question}\n\n"
        f"=== Discussion So Far ===\n{transcript}\n\n"
        "Decide whether the discussion has surfaced enough -- the key considerations, "
        "risks, and design constraints a Planner would need are now covered, and "
        "another round would mostly repeat or marginally refine what's already been "
        "said -- versus whether there is still a substantial, load-bearing gap that "
        "another round would meaningfully close.\n\n"
        "Respond with EXACTLY one character and nothing else: 1 if the meeting should "
        "stop now, 0 if it should continue for another round."
    )
    llm = LLMClient(model=JUDGE_MODEL, provider=JUDGE_PROVIDER, reasoning_effort=JUDGE_REASONING_EFFORT)
    output = llm.complete_text(prompt).strip()
    return {"prompt": prompt, "output": output, "stop": output.startswith("1")}
