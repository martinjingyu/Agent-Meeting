from __future__ import annotations

from typing import Any

from research_agent.llm import LLMClient


def aggregate_responses(
    question: str,
    turns: list[dict[str, Any]],
    strategy: str,
    model: str | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    if strategy == "concat":
        return _aggregate_concat(turns)
    if strategy == "llm":
        return _aggregate_llm(question, turns, model, provider)
    raise ValueError(f"Unknown aggregation strategy: {strategy!r}")


def _aggregate_concat(turns: list[dict[str, Any]]) -> dict[str, Any]:
    output = "\n\n".join(f"[{t['agent']}] {t['output']}" for t in turns)
    return {"strategy": "concat", "prompt": None, "output": output}


def _aggregate_llm(
    question: str,
    turns: list[dict[str, Any]],
    model: str | None,
    provider: str | None,
) -> dict[str, Any]:
    responses_block = "\n\n".join(
        f"### {t['agent']}" + (f" ({t.get('role')})" if t.get("role") else "") + f"\n{t['output']}"
        for t in turns
    )
    prompt = (
        "Multiple participants independently answered the same meeting question. "
        "Synthesize their answers into a single, coherent final response: keep points "
        "they agree on, note meaningful disagreements, and resolve redundancy. Do not "
        "just concatenate their answers.\n\n"
        f"Question:\n{question}\n\n"
        f"Participant answers:\n{responses_block}\n\n"
        "Write the synthesized final response now."
    )
    output = LLMClient(model=model, provider=provider).complete_text(prompt)
    return {"strategy": "llm", "prompt": prompt, "output": output}
