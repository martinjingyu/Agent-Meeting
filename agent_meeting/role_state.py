"""Optional per-meeting isolation for role state (memory.md + workspace/).

Role-backed participants/moderator persist state in roles/<role_ref>/
{memory.md,workspace/} -- shared across every meeting that ever uses that role_ref
(see roles.py). By default (MeetingConfig.persist_role_state=False) a meeting must
not leave any trace there: snapshot_role_states() copies each role's current state
before the meeting's first attempt, and restore_role_states() copies it back once the
meeting finally reaches "completed" status, so the role looks exactly as it did
before the experiment regardless of what participants read/wrote/appended during it.

Both are idempotent/no-ops when there's nothing to do, and restore only ever fires
once per meeting (from runner.run_meeting, right before it returns) -- never on a
resumed sub-call that still ends in failure, since the meeting isn't "done" yet and
wiping workspace files a cached turn already produced would break the next resume.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from . import roles as roles_api
from .config import MeetingConfig
from .storage import RUNS_DIR


def _snapshot_root(meeting_id: str) -> Path:
    return RUNS_DIR / meeting_id / "role_state_snapshot"


def collect_role_refs(config: MeetingConfig) -> set[str]:
    refs = {p.role_ref for p in config.participants if p.role_ref}
    if config.moderator and config.moderator.role_ref:
        refs.add(config.moderator.role_ref)
    return refs


def clear_role_workspaces(role_refs: set[str]) -> None:
    """Delete the contents of each role_ref's persistent workspace/ (never memory.md)
    before a brand-new meeting starts. Defense in depth alongside restore_role_states:
    that only fires once a meeting reaches "completed", so a meeting that crashes or
    gets killed mid-run -- the common case in practice while iterating -- leaves
    whatever it wrote sitting in the role's workspace forever, which is exactly how
    workspaces accumulated dozens of stale .background_*.log/test-script files across
    unrelated past meetings. Called only for a fresh (non-resume) run_meeting() call,
    gated the same way as snapshot/restore_role_states (persist_role_state=False) --
    a resumed call for an already-in-progress meeting must never wipe workspace files
    that meeting's own earlier rounds already produced."""
    for ref in role_refs:
        role = roles_api.load_role(ref)
        if role.workspace_path.exists():
            shutil.rmtree(role.workspace_path)
        role.workspace_path.mkdir(parents=True, exist_ok=True)


def snapshot_role_states(meeting_id: str, role_refs: set[str]) -> None:
    """Idempotent per role: a resumed run_meeting() call for the same meeting_id
    must not re-snapshot over state a prior (failed) attempt already mutated, so a
    role_ref whose snapshot subdir already exists is left untouched."""
    if not role_refs:
        return
    root = _snapshot_root(meeting_id)
    for ref in role_refs:
        role_dir = root / ref
        if role_dir.exists():
            continue
        role_dir.mkdir(parents=True)
        role = roles_api.load_role(ref)
        if role.memory_path and role.memory_path.exists():
            shutil.copy2(role.memory_path, role_dir / "memory.md")
        if role.workspace_path.exists():
            shutil.copytree(role.workspace_path, role_dir / "workspace")


def restore_role_states(meeting_id: str, role_refs: set[str]) -> None:
    """Roll each role's memory.md/workspace back to how it looked right before this
    meeting started, then discard the snapshot."""
    root = _snapshot_root(meeting_id)
    if not role_refs or not root.exists():
        return
    for ref in role_refs:
        role_dir = root / ref
        if not role_dir.exists():
            continue
        role = roles_api.load_role(ref)

        snap_memory = role_dir / "memory.md"
        if role.memory_path:
            if snap_memory.exists():
                shutil.copy2(snap_memory, role.memory_path)
            elif role.memory_path.exists():
                role.memory_path.unlink()

        snap_workspace = role_dir / "workspace"
        if role.workspace_path.exists():
            shutil.rmtree(role.workspace_path)
        if snap_workspace.exists():
            shutil.copytree(snap_workspace, role.workspace_path)

    shutil.rmtree(root, ignore_errors=True)
