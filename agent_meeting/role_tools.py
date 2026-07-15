"""Tool-facing layer over agent_meeting.roles -- NOT loaded by default.

Call register_role_tools() to opt in. Exists so the LLM moderator can autonomously
decide to pull an existing role into a meeting or create a new one; parallel_qa's
deterministic orchestration only needs the plain Python API in agent_meeting/roles.py
and doesn't need to register these.

role_memory is scoped per-run via runtime["role_memory_path"] (set through
GeneralAgent's extra_runtime constructor param) rather than a global override -- safe
across concurrent participant threads, each with their own runtime dict.

Ported from research_agent.tools.roles (removed upstream in Agent-Tutorial commit
bc914f0; see agent_meeting/roles.py's module docstring). Registers into
research_agent's shared global tool registry, same as the original, so
tools_setup.py's registry.without(...) filtering keeps working unchanged.
"""
from __future__ import annotations

from pathlib import Path

from research_agent.md_entries import read_entries, write_entries
from research_agent.tools.registry import json_result, registry

from . import roles as roles_api


def _role_list(args: dict, runtime: dict) -> str:
    names = roles_api.list_roles()
    result = []
    for name in names:
        try:
            role = roles_api.load_role(name)
            result.append({"name": name, "description": role.description})
        except Exception:
            continue
    return json_result(success=True, roles=result, count=len(result))


def _role_load(args: dict, runtime: dict) -> str:
    name = str(args.get("name") or "")
    try:
        role = roles_api.load_role(name)
    except FileNotFoundError as exc:
        return json_result(success=False, error=str(exc))
    return json_result(
        success=True,
        name=role.name,
        description=role.description,
        system_prompt=roles_api.role_system_prompt(role),
    )


def _role_create(args: dict, runtime: dict) -> str:
    name = str(args.get("name") or "")
    description = str(args.get("description") or "")
    body = str(args.get("body") or "")
    frontmatter = {
        k: args.get(k)
        for k in (
            "persona", "purpose", "constraints", "output_contract",
            "style", "stance", "skills", "model", "provider", "max_iterations",
        )
        if args.get(k) is not None
    }
    try:
        role = roles_api.create_role(name, description, body, **frontmatter)
    except (ValueError, FileExistsError) as exc:
        return json_result(success=False, error=str(exc))
    return json_result(success=True, name=role.name)


def _role_memory(args: dict, runtime: dict) -> str:
    path = runtime.get("role_memory_path")
    if not path:
        return json_result(success=False, error="No role_memory_path in runtime for this run")
    path = Path(path)
    action = str(args.get("action") or "read")
    entries = read_entries(path)

    if action == "read":
        return json_result(success=True, entries=entries)
    if action == "add":
        content = str(args.get("content") or "").strip()
        if not content:
            return json_result(success=False, error="content is required")
        if content not in entries:
            entries.append(content)
            write_entries(path, entries)
        return json_result(success=True, message="Entry added")
    if action == "replace":
        old_text = str(args.get("old_text") or "")
        content = str(args.get("content") or "").strip()
        matches = [i for i, entry in enumerate(entries) if old_text and old_text in entry]
        if len(matches) != 1:
            return json_result(success=False, error=f"Expected one match, found {len(matches)}")
        entries[matches[0]] = content
        write_entries(path, entries)
        return json_result(success=True, message="Entry replaced")
    if action == "remove":
        old_text = str(args.get("old_text") or "")
        new_entries = [entry for entry in entries if old_text not in entry]
        removed = len(entries) - len(new_entries)
        write_entries(path, new_entries)
        return json_result(success=True, message="Entries removed", removed=removed)
    return json_result(success=False, error="action must be read, add, replace, or remove")


def register_role_tools() -> None:
    registry.register(
        "role_list",
        {
            "description": "List available roles (name + description) from the role library.",
            "parameters": {"type": "object", "properties": {}},
        },
        _role_list,
    )
    registry.register(
        "role_load",
        {
            "description": "Load a role by name and return its assembled system prompt.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
        _role_load,
    )
    registry.register(
        "role_create",
        {
            "description": (
                "Create a new reusable role. Only name/description are required -- "
                "leave persona/style/stance unset for a purely functional (non-persona) role."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "persona": {"type": "string"},
                    "purpose": {"type": "string"},
                    "constraints": {"type": "array", "items": {"type": "string"}},
                    "output_contract": {"type": "string"},
                    "style": {"type": "string"},
                    "stance": {"type": "string"},
                    "skills": {"type": "array", "items": {"type": "string"}},
                    "model": {"type": "string"},
                    "provider": {"type": "string"},
                    "max_iterations": {"type": "integer"},
                    "body": {"type": "string"},
                },
                "required": ["name", "description"],
            },
        },
        _role_create,
    )
    registry.register(
        "role_memory",
        {
            "description": "Read or update this role's own persistent memory (carries over across meetings).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["read", "add", "replace", "remove"]},
                    "content": {"type": "string"},
                    "old_text": {"type": "string"},
                },
                "required": ["action"],
            },
        },
        _role_memory,
    )
