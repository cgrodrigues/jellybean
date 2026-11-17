# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Skill discovery: turns ``skills/*/SKILL.md`` into agent definitions.

A SKILL.md is YAML front matter followed by Markdown. The front matter holds
``name``, ``description``, ``max_iterations`` (optional) and ``shared_tools``
(the shared tools the agent may use). The Markdown body is the agent's system
prompt. Folders starting with ``_`` (such as ``_shared``) are ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


class SkillFormatError(ValueError):
    """Raised for a malformed SKILL.md."""


class _FrontMatter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    max_iterations: int | None = Field(None, ge=1)
    shared_tools: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class SkillDefinition:
    """An agent definition loaded from disk."""

    name: str
    description: str
    system_prompt: str
    directory: Path
    module: str  # import path of the skill's tools module
    max_iterations: int | None = None
    shared_tools: tuple[str, ...] = field(default_factory=tuple)


def parse_skill_file(path: Path) -> tuple[_FrontMatter, str]:
    """Split a SKILL.md into validated front matter and body."""
    text = path.read_text(encoding="utf-8")
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        raise SkillFormatError(f"{path}: missing YAML front matter delimited by '---'")
    try:
        meta = _FrontMatter.model_validate(yaml.safe_load(match.group(1)) or {})
    except Exception as exc:
        raise SkillFormatError(f"{path}: invalid front matter: {exc}") from exc
    body = match.group(2).strip()
    if not body:
        raise SkillFormatError(f"{path}: empty body (the body is the system prompt)")
    return meta, body


def discover_skills(skills_dir: Path = SKILLS_DIR, package: str = "jellybean.skills") -> dict[str, SkillDefinition]:
    """Load every skill folder that contains a SKILL.md."""
    skills: dict[str, SkillDefinition] = {}
    for folder in sorted(p for p in skills_dir.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))):
        skill_file = folder / "SKILL.md"
        if not skill_file.exists():
            continue
        meta, body = parse_skill_file(skill_file)
        if meta.name != folder.name:
            raise SkillFormatError(f"{skill_file}: name '{meta.name}' must match folder '{folder.name}'")
        skills[meta.name] = SkillDefinition(
            name=meta.name,
            description=meta.description,
            system_prompt=body,
            directory=folder,
            module=f"{package}.{folder.name}.tools",
            max_iterations=meta.max_iterations,
            shared_tools=tuple(meta.shared_tools),
        )
    return skills
