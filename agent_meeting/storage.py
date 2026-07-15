from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"


def new_meeting_id() -> str:
    return f"mtg_{uuid.uuid4().hex[:10]}"


def meeting_path(meeting_id: str) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR / f"{meeting_id}.json"


def sessions_dir(meeting_id: str) -> Path:
    path = RUNS_DIR / meeting_id / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def participant_workspace_dir(meeting_id: str, participant_name: str) -> Path:
    """Private workspace for an ad-hoc (non-role_ref) participant -- ephemeral, tied
    to this one meeting (unlike a role's own persistent roles/<name>/workspace/,
    since an ad-hoc participant has no identity that carries across meetings)."""
    path = RUNS_DIR / meeting_id / "workspace" / participant_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def shared_dir(meeting_id: str) -> Path:
    """The one explicit, opt-in area every participant in this meeting can read AND
    write via the file tools (research_agent's shared_roots mechanism), in addition
    to their own private workspace. Meant for deliberate sharing -- e.g. attaching a
    source document for everyone, or one participant publishing a result for others
    to build on -- never for participants to browse each other's private workspaces
    or leftover files from older meeting runs, which stay off-limits."""
    path = RUNS_DIR / meeting_id / "shared"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_meeting(data: dict[str, Any]) -> Path:
    path = meeting_path(data["meeting_id"])
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def load_meeting(meeting_id: str) -> dict[str, Any]:
    path = meeting_path(meeting_id)
    return json.loads(path.read_text(encoding="utf-8"))


def meeting_exists(meeting_id: str) -> bool:
    return meeting_path(meeting_id).exists()


# ── Per-turn cache (resume support) ────────────────────────────────────────
# A completed participant turn is cached the moment it finishes, independent of the
# round-level checkpoint in save_meeting(). This lets a resume skip re-running
# participants that already finished within a round that later failed partway
# through (e.g. one participant hit a transient API error while others had already
# completed) -- not just skip whole already-completed rounds.

def turn_cache_dir(meeting_id: str) -> Path:
    path = RUNS_DIR / meeting_id / "turn_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def turn_cache_path(meeting_id: str, round_num: int, participant_name: str) -> Path:
    return turn_cache_dir(meeting_id) / f"r{round_num}_{participant_name}.json"


def save_turn_cache(meeting_id: str, round_num: int, participant_name: str, turn: dict[str, Any]) -> None:
    path = turn_cache_path(meeting_id, round_num, participant_name)
    path.write_text(json.dumps(turn, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_turn_cache(meeting_id: str, round_num: int, participant_name: str) -> dict[str, Any] | None:
    path = turn_cache_path(meeting_id, round_num, participant_name)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
