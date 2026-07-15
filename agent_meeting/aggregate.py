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
    if strategy == "audited_llm":
        return _aggregate_audited_llm(question, turns, model, provider)
    if strategy == "final_audit_llm":
        return _aggregate_final_audit_llm(question, turns, model, provider)
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


def _responses_block(turns: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"### {t['agent']}" + (f" ({t.get('role')})" if t.get("role") else "") + f"\n{t['output']}"
        for t in turns
    )


def _aggregate_audited_llm(
    question: str,
    turns: list[dict[str, Any]],
    model: str | None,
    provider: str | None,
) -> dict[str, Any]:
    responses_block = _responses_block(turns)
    prompt = (
        "Multiple participants answered the same planning question. Synthesize them, "
        "but do constraint audit before consensus. Agreement is not evidence, and an "
        "idea repeated by many participants must still be rejected if it violates the "
        "original task constraints.\n\n"
        "Audit rules:\n"
        "1. First extract non-negotiable constraints from the question.\n"
        "2. Reject or quarantine any proposal that uses forbidden information, "
        "hardcodes current datasets, overfits small validation samples, exceeds runtime "
        "or dependency constraints, or turns an observation into an unsupported rule.\n"
        "3. Classify evidence as: HARD_CONSTRAINT, MEASURED_VISUAL_EVIDENCE, "
        "CURRENT_DATA_OBSERVATION, SPECULATION, or FORBIDDEN_NON_VISUAL_PRIOR.\n"
        "4. Treat dataset names, file names, paths, source notes, timestamps, EXIF, and "
        "directory structure as allowed only for traversal, grouping, cost planning, "
        "traceability, and reporting unless the question explicitly allows them for "
        "the target judgment.\n"
        "5. Do not write 'all agree' unless every participant explicitly agrees and the "
        "point survives the audit. If a point is popular but invalid, list it under "
        "Rejected / do not implement.\n"
        "6. Produce an executor-ready answer with clear accepted decisions, rejected "
        "decisions, unresolved risks, and implementation handoff notes.\n\n"
        f"Question:\n{question}\n\n"
        f"Participant answers:\n{responses_block}\n\n"
        "Write the audited synthesized response now."
    )
    output = LLMClient(model=model, provider=provider).complete_text(prompt)
    return {"strategy": "audited_llm", "prompt": prompt, "output": output}


def _aggregate_final_audit_llm(
    question: str,
    turns: list[dict[str, Any]],
    model: str | None,
    provider: str | None,
) -> dict[str, Any]:
    draft = _responses_block(turns)
    prompt = (
        "You are the final compliance and evidence auditor for a multi-agent planning "
        "meeting. Your job is to repair the draft into a safe final answer, not to "
        "preserve consensus.\n\n"
        "Final audit checklist:\n"
        "- Remove or explicitly reject any use of forbidden non-visual information for "
        "the target judgment.\n"
        "- Remove or quarantine dataset-specific shortcuts unless they are framed only "
        "as configurable cost/risk hints and not suitability rules.\n"
        "- Downgrade small-sample findings to configurable defaults or calibration "
        "inputs, not universal facts.\n"
        "- Keep unsupported performance numbers out of hard guarantees.\n"
        "- Preserve traceability, conservative selection, degraded modes, QA outputs, "
        "and known limitations.\n"
        "- Include a short 'Rejected by audit' section when the draft contained "
        "plausible but invalid ideas.\n\n"
        f"Original question:\n{question}\n\n"
        f"Draft to audit:\n{draft}\n\n"
        "Write the final audited answer now."
    )
    output = LLMClient(model=model, provider=provider).complete_text(prompt)
    return {"strategy": "final_audit_llm", "prompt": prompt, "output": output}
