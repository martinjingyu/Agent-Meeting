"""Role management: reusable participant identities stored as roles/<name>/ folders.

A role is NOT a persona by default -- DEFINITION.md's frontmatter fields are all
optional except name/description, so a purely functional agent ("you are a thing that
extracts risk factors") is just as valid as a personified one ("you are Jordan, a
jaded VC"). See DEFINITION.md schema in role_system_prompt()'s field order below.

Ported from research_agent.roles (removed upstream in Agent-Tutorial commit bc914f0,
which repurposed research_agent/roles.py for an unrelated concept -- the agent's own
operating-mode prompt profiles in research_agent/prompts/roles.py's ROLE_PROFILES).
Agent-Meeting is the only consumer of the persona-role concept, so it now owns this
module outright; storage root and the underlying entry-list format still come from
research_agent (roles_root()/set_roles_root() in paths.py, read_entries/write_entries
in md_entries.py), which weren't part of the deletion.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from research_agent.md_entries import read_entries, write_entries
from research_agent.paths import roles_root
from research_agent.tools.skills import _find_skill

VALID_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

# Order controls how role_system_prompt() renders the assembled prompt.
_PROMPT_FIELDS: list[tuple[str, str]] = [
    ("persona", ""),
    ("purpose", "Purpose"),
    ("output_contract", "Output contract"),
    ("style", "Style"),
    ("stance", "Stance toward other participants"),
]


@dataclass
class RoleDefinition:
    name: str
    frontmatter: dict[str, Any] = field(default_factory=dict)
    body: str = ""
    memory_path: Path | None = None

    @property
    def description(self) -> str:
        return str(self.frontmatter.get("description") or "")

    @property
    def skill_names(self) -> list[str]:
        skills = self.frontmatter.get("skills") or []
        return [str(s) for s in skills] if isinstance(skills, list) else []

    @property
    def model(self) -> str | None:
        return self.frontmatter.get("model")

    @property
    def provider(self) -> str | None:
        return self.frontmatter.get("provider")

    @property
    def reasoning_effort(self) -> str | None:
        return self.frontmatter.get("reasoning_effort")

    @property
    def max_iterations(self) -> int:
        return int(self.frontmatter.get("max_iterations") or 8)

    @property
    def workspace_path(self) -> Path:
        return _role_dir(self.name) / "workspace"


_VERDICT_WORD_RE = re.compile(r"\b([A-Za-z]{3,12})\b")
_EXPLICIT_VERDICT_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:[-*]\s*)?(?:\*\*)?\s*VERDICT\s*:?\s*(?:\*\*)?\s*([A-Za-z]{3,12})\b"
)
# A verdict heading on its own line (e.g. markdown "## Verdict"), with nothing
# else on that line -- the actual token is expected on one of the following
# lines (e.g. "## Verdict\n\n**REVISE** -- ..."), which _EXPLICIT_VERDICT_RE
# alone can't reach since it only looks at the same line as "VERDICT".
_VERDICT_HEADING_RE = re.compile(r"(?im)^\s*#{1,6}\s*verdict\s*:?\s*$")
# A token that leads a line (after stripping bullet/bold markers), e.g.
# "**REVISE** -- three material corrections" or "REVISE: ...". Deliberately
# anchored to the start of the line -- unlike _VERDICT_WORD_RE's bare \b match,
# this can't be tricked by a token word used in prose later in the same line
# (see module-level extract_output_contract_verdict docstring for why that
# matters).
_LEADING_TOKEN_RE = re.compile(r"^(?:[-*]\s*)?(?:\*\*)?\s*([A-Za-z]{3,12})\b")


def extract_output_contract_verdict(role: "RoleDefinition", output: str) -> str | None:
    """If role.output_contract declares a pipe-separated enum of verdict tokens
    (e.g. skeptic-reviewer's 'ACCEPT | REVISE | REJECT'), return the verdict token
    stated in the participant's actual turn output, or None if the contract doesn't
    declare such an enum or the output has no unambiguous verdict.

    This exists so a role's stated verdict (e.g. Skeptic writing "VERDICT: REVISE")
    is available as structured data instead of only living as free text buried in
    the transcript the judge reads -- a judge LLM can rationalize past a sentence
    it disagrees with, but callers can check this field deterministically and,
    e.g., refuse to let the meeting stop while a REVISE/REJECT verdict stands.
    The parser deliberately prefers an explicit line such as "VERDICT: REVISE".
    If no such line exists, it only falls back to a verdict token that LEADS one
    of the final lines (after stripping bullet/bold markers). It must not scan
    the whole output -- or even a whole candidate line -- for any occurrence of
    an enum word, because explanatory prose routinely uses those same words with
    their ordinary English meaning (e.g. "the detection ceiling is now fully
    measured -- accept it and design around X" is not a verdict declaration, it's
    a sentence that happens to contain the word "accept"; a bare \\b-word scan
    over the whole line matched exactly that and returned ACCEPT instead of the
    real "**REVISE**" verdict two lines above it -- see roles_test / the R9
    Skeptic turn in mtg_6b0c464bc9 for the real transcript that exposed this)."""
    contract = str(role.frontmatter.get("output_contract") or "")
    tokens = set(re.findall(r"[A-Z]{3,12}", contract))
    if len(tokens) < 2:
        return None

    explicit_matches = [
        match.group(1).upper()
        for match in _EXPLICIT_VERDICT_RE.finditer(output)
        if match.group(1).upper() in tokens
    ]
    if explicit_matches:
        return explicit_matches[-1]

    lines = output.splitlines()
    for i, line in enumerate(lines):
        if not _VERDICT_HEADING_RE.match(line):
            continue
        for follower in lines[i + 1:i + 4]:
            follower = follower.strip()
            if not follower:
                continue
            match = _LEADING_TOKEN_RE.match(follower)
            if match and match.group(1).upper() in tokens:
                return match.group(1).upper()
            break  # first non-blank line after the heading didn't lead with a token

    nonempty_lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in reversed(nonempty_lines[-5:]):
        match = _LEADING_TOKEN_RE.match(line)
        if match and match.group(1).upper() in tokens:
            return match.group(1).upper()
    return None


def _role_dir(name: str) -> Path:
    return roles_root() / name


def _definition_path(name: str) -> Path:
    return _role_dir(name) / "DEFINITION.md"


def _memory_path(name: str) -> Path:
    return _role_dir(name) / "memory.md"


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    match = re.search(r"\n---\s*\n", text[3:])
    if not match:
        return {}, text
    end = match.start() + 3
    data = yaml.safe_load(text[3:end]) or {}
    body = text[match.end() + 3:]
    return (data if isinstance(data, dict) else {}), body


def _validate_name(name: str) -> str | None:
    if not name:
        return "name is required"
    if len(name) > 64 or not VALID_NAME_RE.match(name):
        return "name must be lowercase letters/numbers plus . _ -, starting with a letter or digit"
    return None


def list_roles() -> list[str]:
    root = roles_root()
    if not root.exists():
        return []
    return sorted(
        p.parent.name for p in root.glob("*/DEFINITION.md")
    )


def load_role(name: str) -> RoleDefinition:
    path = _definition_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Role not found: {name} (looked in {path})")
    frontmatter, body = _parse_frontmatter(path.read_text(encoding="utf-8", errors="replace"))
    return RoleDefinition(
        name=name,
        frontmatter=frontmatter,
        body=body.strip(),
        memory_path=_memory_path(name),
    )


def create_role(
    name: str,
    description: str,
    body: str = "",
    *,
    overwrite: bool = False,
    **frontmatter: Any,
) -> RoleDefinition:
    error = _validate_name(name)
    if error:
        raise ValueError(error)
    if not description.strip():
        raise ValueError("description is required")
    path = _definition_path(name)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Role already exists: {name} (pass overwrite=True to replace)")

    data = {"name": name, "description": description.strip(), **frontmatter}
    # Drop empty/None fields so the file only shows what was actually set --
    # matches the taxonomy: omitted fields mean "not applicable to this role",
    # not "explicitly blank".
    data = {k: v for k, v in data.items() if v not in (None, "", [], {})}

    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter_text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False).strip()
    path.write_text(f"---\n{frontmatter_text}\n---\n{body.strip()}\n", encoding="utf-8")

    return load_role(name)


def role_memory_entries(role: RoleDefinition) -> list[str]:
    if role.memory_path is None:
        return []
    return read_entries(role.memory_path)


def append_role_memory(role: RoleDefinition, content: str) -> None:
    if role.memory_path is None:
        return
    entries = read_entries(role.memory_path)
    if content not in entries:
        entries.append(content)
        write_entries(role.memory_path, entries)


def role_system_prompt(role: RoleDefinition, include_memory: bool = True) -> str:
    parts: list[str] = [f"You are {role.name}."]

    for key, label in _PROMPT_FIELDS:
        value = role.frontmatter.get(key)
        if not value:
            continue
        parts.append(f"{label}: {value}" if label else str(value))

    constraints = role.frontmatter.get("constraints")
    if constraints:
        lines = "\n".join(f"- {c}" for c in constraints)
        parts.append(f"Constraints:\n{lines}")

    if role.body:
        parts.append(role.body)

    if role.skill_names:
        lines = []
        for skill_name in role.skill_names:
            skill_dir = _find_skill(skill_name)
            if skill_dir is None:
                continue
            skill_md = skill_dir / "SKILL.md"
            frontmatter, _ = _parse_frontmatter(skill_md.read_text(encoding="utf-8", errors="replace"))
            desc = str(frontmatter.get("description") or "").strip()
            lines.append(f"- {skill_name}" + (f" — {desc}" if desc else ""))
        if lines:
            parts.append(
                "Assigned skills (use skill_view to read full details):\n" + "\n".join(lines)
            )

    if include_memory:
        entries = role_memory_entries(role)
        if entries:
            lines = "\n".join(f"- {e}" for e in entries)
            parts.append(f"[Persistent role memory]\n{lines}")

    return "\n\n".join(parts)
