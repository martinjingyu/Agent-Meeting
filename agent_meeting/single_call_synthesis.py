"""Shared streaming-with-continuation logic for synthesizing a long document (a
final Plan, in every current caller) with a single LLM call instead of an agentic
tool loop. Used by both examples/synthesize_final_plan_single_call.py (standalone,
reads a meeting checkpoint off disk after the fact) and runner.py's
_run_planner_step (PlannerConfig.synthesis="single_call", wired straight into a
normal meeting run) -- kept in one place so the two don't drift out of sync.

Streaming, not a single blocking call: a big prompt (task + full multi-round
transcript) at high reasoning effort can go quiet for a long stretch before its
first visible token, which is indistinguishable from a hang on a plain blocking
call. Printing each delta as it arrives (when verbose) makes real progress visible.

Resilience loop, not raw chat_stream(): chat_stream() itself has no automatic
retry (see its docstring in research_agent/llm.py -- retrying a partially-yielded
stream would duplicate content). This module covers two distinct failure shapes on
top of it:
  - A connection drop before any delta of the current segment arrived: safe to just
    retry the identical call (nothing to lose or duplicate).
  - A connection drop after some content already streamed: that content is kept
    as-is, and the conversation is extended with an assistant turn holding exactly
    what arrived plus a user turn asking the model to continue from there -- then a
    fresh chat_stream() call picks up where the last one was cut off. This repeats
    until a segment completes without an exception, and every segment's text is
    concatenated at the end. The seam between two segments isn't guaranteed
    byte-perfect, but no content is lost the way a plain retry-from-scratch would.
"""
from __future__ import annotations

import time

from research_agent.llm import LLMClient

_MAX_ZERO_PROGRESS_RETRIES = 5
_RETRY_SLEEP_SECONDS = 5.0
_MAX_CONTINUATIONS = 20
_COMPLETE_MARKER = "<<PLAN_COMPLETE>>"
_CONTINUE_PROMPT = (
    "Your previous response was cut off mid-stream by a connection issue, not because "
    "you had finished. Continue EXACTLY where you left off -- do not repeat anything "
    "already written, do not restart, do not add a new heading or preamble, do not "
    "re-summarize. Resume with the very next word/sentence/section as if there had "
    f"been no interruption. If you had, in fact, already finished the entire document, "
    f"respond with exactly: {_COMPLETE_MARKER}"
)


def stream_synthesize(
    system_prompt: str,
    user_message: str,
    *,
    model: str,
    provider: str,
    reasoning_effort: str | None,
    verbose: bool = True,
    label: str = "synthesize",
) -> str:
    """One document, one LLM call (streamed, with resilience) -- no tools, no agent
    loop, no file access. Raises RuntimeError if the model never produces usable
    content after exhausting retries/continuations; callers should let that
    propagate rather than silently returning a partial/empty document."""
    llm = LLMClient(model=model, provider=provider, reasoning_effort=reasoning_effort)
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    full_parts: list[str] = []
    zero_progress_retries = 0
    continuations = 0
    while True:
        segment_parts: list[str] = []
        interrupted = False
        try:
            for delta in llm.chat_stream(messages, tools=[]):
                if verbose:
                    print(delta, end="", flush=True)
                segment_parts.append(delta)
        except Exception as exc:
            interrupted = True
            if not segment_parts:
                zero_progress_retries += 1
                if zero_progress_retries > _MAX_ZERO_PROGRESS_RETRIES:
                    raise
                if verbose:
                    print(
                        f"\n[{label}] connection failed before any content arrived "
                        f"({type(exc).__name__}); retrying "
                        f"({zero_progress_retries}/{_MAX_ZERO_PROGRESS_RETRIES})...",
                        flush=True,
                    )
                time.sleep(_RETRY_SLEEP_SECONDS)
                continue
            zero_progress_retries = 0

        segment_text = "".join(segment_parts)
        if segment_text.strip() != _COMPLETE_MARKER:
            full_parts.append(segment_text)

        if not interrupted:
            break  # this segment finished on its own -- the document is done

        continuations += 1
        if continuations > _MAX_CONTINUATIONS:
            raise RuntimeError(
                f"[{label}] gave up after {_MAX_CONTINUATIONS} continuations without "
                "the model finishing -- the connection may be too unstable right now "
                "for a request this size."
            )
        if verbose:
            print(
                f"\n[{label}] connection dropped mid-response after "
                f"{len(segment_text):,} chars; requesting a continuation "
                f"({continuations}/{_MAX_CONTINUATIONS})...",
                flush=True,
            )
        messages.append({"role": "assistant", "content": segment_text})
        messages.append({"role": "user", "content": _CONTINUE_PROMPT})

    if verbose:
        print()
    document = "".join(full_parts)
    if not document.strip():
        raise RuntimeError(f"[{label}] {model} returned an empty response.")
    return document
