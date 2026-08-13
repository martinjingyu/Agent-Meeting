from pathlib import Path

from research_agent.paths import set_roles_root

# Agent-Meeting keeps its own private role library instead of writing into
# Agent-Tutorial's own roles/ directory (same reasoning as this project's runs/
# directory being separate from Agent-Tutorial/sessions/).
set_roles_root(Path(__file__).resolve().parents[1] / "roles")

# Side effect: raises research_agent's compaction trigger for every GeneralAgent
# this package builds -- see _context_limits.py for why.
from ._context_limits import COMPACT_TOKEN_THRESHOLD

from .config import MeetingConfig, ModeratorConfig, ParticipantConfig
from .runner import run_meeting

__all__ = [
    "COMPACT_TOKEN_THRESHOLD",
    "MeetingConfig",
    "ModeratorConfig",
    "ParticipantConfig",
    "run_meeting",
]
