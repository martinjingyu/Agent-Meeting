"""Builds the tool registry participant agents run with: everything research_agent
ships except kanban (kanban is a task-board/notification system tied to Agent-Tutorial's
own workflows and has no meaning inside a meeting participant's turn).
"""
from __future__ import annotations

from research_agent.tools import load_builtin_tools, registry
from research_agent.tools.registry import ToolRegistry
from research_agent.tools.roles import register_role_tools

from .round_tools import register_round_tools

KANBAN_TOOL_NAMES: set[str] = {
    "kanban_create_task",
    "kanban_list_tasks",
    "kanban_update_task",
    "kanban_dispatch",
    "kanban_notify_subscribe",
    "kanban_create_pipeline",
    "kanban_create_meeting_task",
}

# role_list/role_load/role_create are moderator-oriented (browsing/creating roles at
# large); a role-backed participant only gets role_memory (read/update its own
# persistent memory) so it can't spawn new roles or read others' definitions mid-turn.
ROLE_MANAGEMENT_TOOL_NAMES: set[str] = {"role_list", "role_load", "role_create"}


def build_participant_registry(role_backed: bool = False, round_aware: bool = False) -> ToolRegistry:
    """round_aware=True swaps respond_to_user out for submit_round_answer (round >= 2
    finish tool, required changes_from_prior_round field) -- round 1 always finishes
    via the normal respond_to_user."""
    load_builtin_tools()
    excluded = set(KANBAN_TOOL_NAMES)
    if role_backed:
        register_role_tools()
        excluded |= ROLE_MANAGEMENT_TOOL_NAMES
    else:
        excluded |= ROLE_MANAGEMENT_TOOL_NAMES | {"role_memory"}
    if round_aware:
        register_round_tools()
        excluded |= {"respond_to_user"}
    return registry.without(excluded)
