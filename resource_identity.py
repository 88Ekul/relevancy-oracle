"""Stable, source-qualified identities for Oracle resources."""
from __future__ import annotations

import re


_SOURCES = {"github": {"repository"}, "huggingface": {"model", "dataset", "space"}}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def resource_identity(source: str, repo_type: str, identifier: str) -> str:
    """Return a collision-proof identity such as ``huggingface:model:a/b``."""
    source = str(source).lower().strip()
    repo_type = str(repo_type).lower().strip()
    identifier = str(identifier).strip()
    if source not in _SOURCES or repo_type not in _SOURCES[source]:
        raise ValueError("Unsupported resource source or repository type.")
    if not _IDENTIFIER.fullmatch(identifier) or any(part in (".", "..") for part in identifier.split("/")):
        raise ValueError("Resource identifier must be an owner/name pair.")
    return f"{source}:{repo_type}:{identifier}"


def parse_resource_identity(value: str) -> dict:
    """Validate and split a source-qualified identity."""
    if not isinstance(value, str):
        raise ValueError("Resource identity must be a string.")
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise ValueError("Resource identity must contain source, type and identifier.")
    source, repo_type, identifier = parts
    expected = resource_identity(source, repo_type, identifier)
    return {"identity": expected, "source": source, "repo_type": repo_type,
            "identifier": identifier}
