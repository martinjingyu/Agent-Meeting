"""Builds the tool registry participant agents run with: everything research_agent
ships except kanban (kanban is a task-board/notification system tied to Agent-Tutorial's
own workflows and has no meaning inside a meeting participant's turn).
"""
from __future__ import annotations

from research_agent.tools import load_builtin_tools, registry
from research_agent.tools.registry import ToolRegistry

from .role_tools import register_role_tools
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

# run_background/check_background hand a job off to a watcher thread and expect the
# caller to come back on a *later turn of the same session* to receive the completion
# notification (consume_pending_background_notifications(), called at the top of every
# GeneralAgent.run()). Every Agent-Meeting participant turn is a fresh, one-shot
# GeneralAgent with no later turn in that same session to deliver it to -- and the
# notification directory isn't scoped per participant, so it could just as easily land
# in a different participant's next turn. respond_to_user also has no has_active_jobs()
# check, so a participant that starts a background job and (as run_background's own
# tool description instructs) immediately calls respond_to_user ends its turn with a
# placeholder "waiting on job" answer while the real result is silently lost or
# misdelivered. `terminal` stays available and simply blocks for the command's full
# duration instead of backgrounding it.
BACKGROUND_TOOL_NAMES: set[str] = {"run_background", "check_background"}

# role_list/role_load/role_create are moderator-oriented (browsing/creating roles at
# large); a role-backed participant only gets role_memory (read/update its own
# persistent memory) so it can't spawn new roles or read others' definitions mid-turn.
ROLE_MANAGEMENT_TOOL_NAMES: set[str] = {"role_list", "role_load", "role_create"}


def build_participant_registry(role_backed: bool = False, round_aware: bool = False) -> ToolRegistry:
    """round_aware=True swaps respond_to_user out for submit_round_answer (round >= 2
    finish tool, required changes_from_prior_round field) -- round 1 always finishes
    via the normal respond_to_user."""
    load_builtin_tools()
    excluded = set(KANBAN_TOOL_NAMES) | set(BACKGROUND_TOOL_NAMES)
    if role_backed:
        register_role_tools()
        excluded |= ROLE_MANAGEMENT_TOOL_NAMES
    else:
        excluded |= ROLE_MANAGEMENT_TOOL_NAMES | {"role_memory"}
    if round_aware:
        register_round_tools()
        excluded |= {"respond_to_user"}
    return registry.without(excluded)
