"""Meeting turns run on much larger prompts than research_agent's own defaults
assume -- a single round_message can run into six figures of tokens (task spec +
Stage-1 evidence + accumulated discussion transcript) before a turn's own tool-call
loop even starts. Left at research_agent's defaults (~180,000 chars / 90,000
tokens), compaction would start eating a turn's own freshly-delivered context
before it had a real chance to act on it. This raises the trigger to 800,000
tokens for every agent this package builds.

Imported for its module-level side effect by agent_meeting/__init__.py, so the
override is applied as soon as this package is imported -- before any GeneralAgent
is constructed.
"""
import research_agent.agent as _research_agent

COMPACT_TOKEN_THRESHOLD = 800_000
"""Passed explicitly as context_threshold_tokens at every GeneralAgent(...) call
site in this package (research_agent.agent.GeneralAgent's token-based compaction
check is a per-instance constructor parameter, so this half of the override just
needs to be threaded through as an ordinary argument)."""

# TRAJECTORY_COMPRESS_THRESHOLD (research_agent's *character*-based compaction
# check) is a hardcoded module-level constant, not a constructor parameter, so
# raising it without editing research_agent's own source means overriding the
# module attribute here instead. The check that reads it
# (GeneralAgent's main loop, "over_chars = ... >= TRAJECTORY_COMPRESS_THRESHOLD")
# looks it up as a free variable off research_agent.agent's namespace at call
# time, so this one assignment covers every GeneralAgent instance built in this
# process for as long as agent_meeting stays imported -- it is a process-wide
# override, not scoped to any one agent.
_research_agent.TRAJECTORY_COMPRESS_THRESHOLD = COMPACT_TOKEN_THRESHOLD * 4
