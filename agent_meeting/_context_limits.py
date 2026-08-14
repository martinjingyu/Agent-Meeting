"""Meeting turns run on much larger prompts than research_agent's own defaults
assume -- a single round_message can run into six figures of tokens (task spec +
Stage-1 evidence + accumulated discussion transcript) before a turn's own tool-call
loop even starts. Left at research_agent's defaults (~180,000 chars / 90,000
tokens), compaction would start eating a turn's own freshly-delivered context
before it had a real chance to act on it.

Imported for its module-level side effect by agent_meeting/__init__.py, so the
override is applied as soon as this package is imported -- before any GeneralAgent
is constructed.
"""
import research_agent.agent as _research_agent

AUTO_COMPACT = False
"""Passed explicitly as auto_compact at every GeneralAgent(...) call site in this
package. research_agent gates BOTH of its compaction triggers behind this one
per-instance constructor flag -- the pre-action check (tool-result count + 60% of
context_threshold_tokens, agent.py's _pre_action_compact_check) and the main-loop
gatekeeper (the "trajectory exceeded"/"context threshold exceeded" char/token
check) both start with `if self.auto_compact and ...`. So auto_compact=False turns
both off in one shot, entirely from this package -- no research_agent edit needed."""

COMPACT_TOKEN_THRESHOLD = 800_000
"""Passed explicitly as context_threshold_tokens at every GeneralAgent(...) call
site in this package, in case auto_compact is ever turned back on for some agent
(e.g. debugging) -- a sane fallback so re-enabling it doesn't silently revert to
research_agent's much smaller default."""

# TRAJECTORY_COMPRESS_THRESHOLD (research_agent's *character*-based compaction
# check) is a hardcoded module-level constant, not a constructor parameter, so
# raising it without editing research_agent's own source means overriding the
# module attribute here instead. The check that reads it
# (GeneralAgent's main loop, "over_chars = ... >= TRAJECTORY_COMPRESS_THRESHOLD")
# looks it up as a free variable off research_agent.agent's namespace at call
# time, so this one assignment covers every GeneralAgent instance built in this
# process for as long as agent_meeting stays imported -- it is a process-wide
# override, not scoped to any one agent. Same "fallback for if auto_compact gets
# re-enabled somewhere" reasoning as COMPACT_TOKEN_THRESHOLD above.
_research_agent.TRAJECTORY_COMPRESS_THRESHOLD = COMPACT_TOKEN_THRESHOLD * 4
