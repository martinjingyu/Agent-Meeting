from pathlib import Path

from research_agent.paths import set_roles_root

# Agent-Meeting keeps its own private role library instead of writing into
# Agent-Tutorial's own roles/ directory (same reasoning as this project's runs/
# directory being separate from Agent-Tutorial/sessions/).
set_roles_root(Path(__file__).resolve().parents[1] / "roles")

from .config import MeetingConfig, ModeratorConfig, ParticipantConfig
from .runner import run_meeting

__all__ = ["MeetingConfig", "ModeratorConfig", "ParticipantConfig", "run_meeting"]
