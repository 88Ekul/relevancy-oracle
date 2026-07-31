"""relevancy_oracle.py — the combined relevancy oracle.

A thin orchestrator that runs both halves of the oracle and returns one
combined answer to: "I want to build X — what do I already have, and
what do I need?"

    inward_check  -> what you already own (starred repos that match)
    outward_check -> what you're missing (vetted candidates for the gaps)

The integration that makes the combined run smarter than either half
alone: inward runs first, and its matches are handed to outward so
outward excludes repos you already own and recommends only genuine gaps.

Stateless. JSON-clean in, JSON-clean out. Imports nothing from
Second-Brain-OS — the dependency runs one way only.

Contract: INTERFACE_CONTRACT.md v0.1.

CLI:
    python relevancy_oracle.py "I want to build a WhatsApp CRM" personal
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from inward_check import inward_check
from outward_check import outward_check

ORACLE_VERSION = "0.1"

_VALID_USAGE = ("personal", "commercial")

_USAGE = (
    'Usage: python relevancy_oracle.py "I want to build X" '
    "[personal|commercial]"
)


def _normalise_request(request: dict) -> dict:
    """Validate and fill out the four-field request.

    Only ``query`` and ``usage`` are required. ``audience`` and
    ``stack_hints`` default to None / empty list, so callers (and the
    CLI) may omit them.
    """
    if not isinstance(request, dict):
        raise TypeError("request must be a dict")

    query = request.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("request['query'] must be a non-empty string")

    usage = request.get("usage")
    if usage not in _VALID_USAGE:
        raise ValueError(
            f"request['usage'] must be one of {_VALID_USAGE}, got {usage!r}"
        )

    stack_hints = request.get("stack_hints")
    if stack_hints is None:
        stack_hints = []

    return {
        "query": query.strip(),
        "usage": usage,
        "audience": request.get("audience"),
        "stack_hints": list(stack_hints),
    }


def relevancy_oracle(request: dict) -> dict:
    """Run both halves of the oracle and assemble the combined envelope.

    Args:
        request: the four-field request dict. ``query`` (str) and
            ``usage`` ("personal" | "commercial") are required;
            ``audience`` (str | None) and ``stack_hints``
            (list[str] | None) are optional.

    Returns:
        The combined envelope — inward and outward blocks together, with
        ``brief`` left null. The human-readable brief is rendered
        separately (render_brief over this data), never in the core.
    """
    req = _normalise_request(request)

    # 1. Inward first — what do I already own? Runs on the query alone.
    inward = inward_check(req)

    # 2. Outward second — hand it inward's matches so it excludes repos
    #    already owned and recommends only the genuine gaps. This is the
    #    integration: the combined run beats either half in isolation.
    inward_matches = inward.get("matches", []) if isinstance(inward, dict) else []
    outward = outward_check(req, inward_matches=inward_matches)

    # 3. Assemble the contract envelope. `brief` stays null by design.
    return {
        "query": req["query"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "oracle_version": ORACLE_VERSION,
        "inward": inward,
        "outward": outward,
        "brief": None,
    }


def _cli(argv: list[str]) -> int:
    """Thin CLI: query required, usage optional (defaults to personal)."""
    if argv[:1] in (["-h"], ["--help"]):
        print(_USAGE)
        return 0
    if not argv:
        print(_USAGE, file=sys.stderr)
        return 2

    query = argv[0]
    usage = argv[1] if len(argv) > 1 else "personal"

    try:
        result = relevancy_oracle({"query": query, "usage": usage})
    except (ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
