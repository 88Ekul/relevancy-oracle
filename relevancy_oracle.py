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
import argparse
import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from inward_check import inward_check
from outward_check import outward_check
from assembly_advice import AssemblyAdviser
from huggingface_provider import HuggingFaceProvider
from source_inspection import SourceInspector

ORACLE_VERSION = "0.3"

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

    hf_imports = request.get("hf_imports") or []
    if not isinstance(hf_imports, list) or not all(isinstance(x, str) for x in hf_imports):
        raise ValueError("request['hf_imports'] must be a list of URLs")
    hf_username = request.get("hf_username")
    if hf_username is not None and not isinstance(hf_username, str):
        raise ValueError("request['hf_username'] must be a string or null")
    return {
        "query": query.strip(),
        "usage": usage,
        "audience": request.get("audience"),
        "stack_hints": list(stack_hints),
        "hf_username": hf_username.strip() if isinstance(hf_username, str) else None,
        "hf_imports": hf_imports[:10],
        "refresh_hf_catalogue": bool(request.get("refresh_hf_catalogue")),
    }


def relevancy_oracle(request: dict, inspector: SourceInspector | None = None,
                     hf_provider: HuggingFaceProvider | None = None,
                     assembler: AssemblyAdviser | None = None) -> dict:
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
    inspector = inspector or SourceInspector()
    inspector.inspect_inward(inward_matches, req)
    outward = outward_check(req, inward_matches=inward_matches, inspector=inspector)
    uninspected = sum(m['source_assessment']['status'] != 'assessed' for m in inward_matches)
    if uninspected:
        outward.setdefault('gaps_unfilled', []).append(
            f'{uninspected} catalogue suggestions have no completed source assessment; their contribution is unconfirmed.')
    outward.setdefault('gaps_unfilled', []).append(
        'The assembled application and proposed adaptations have not been executed or integration-tested.')

    # 3. Hugging Face is a separate provider and catalogue. It never receives
    # GitHub credentials and never writes to the GitHub stars catalogue.
    try:
        hf_request = {**req, 'known_gaps': list(outward.get('identified_gaps', []))}
        huggingface = (hf_provider or HuggingFaceProvider(username=req.get('hf_username'))).collect(hf_request)
    except Exception as exc:
        huggingface = {'status': 'unavailable', 'provider': 'huggingface',
                       'catalogue': {'status': 'unavailable', 'resources': []},
                       'public_search': {}, 'imports': [], 'assessments': [],
                       'limitations': [f'Hugging Face provider failed ({type(exc).__name__}); resources remain unresolved.'],
                       'execution': 'not_run'}
    if huggingface.get('status') == 'unavailable':
        outward.setdefault('gaps_unfilled', []).append(
            'Hugging Face discovery was unavailable; model, dataset and Space options remain unresolved.')
    elif huggingface.get('status') == 'no_matches':
        outward.setdefault('gaps_unfilled', []).append(
            'Hugging Face searches completed but returned no matches for the planned ML capabilities.')

    # 4. Cross-source advice may select only validated inspected resources.
    outward_items = outward.get('source_reviews') or outward.get('candidates') or []
    assembly = (assembler or AssemblyAdviser()).assemble(
        req, inward_matches, outward_items, huggingface, outward.get('identified_gaps', []))

    # 5. Assemble the contract envelope. `brief` stays null by design.
    return {
        "query": req["query"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "oracle_version": ORACLE_VERSION,
        "inward": inward,
        "outward": outward,
        "source_inspection": inspector.report(),
        "huggingface": huggingface,
        "assembly": assembly,
        "brief": None,
    }


def _cli(argv: list[str]) -> int:
    """Save the envelope before rendering so a delivery retry costs no API calls."""
    parser = argparse.ArgumentParser(description='Inspect GitHub and Hugging Face resources and assemble build advice.')
    parser.add_argument('query', nargs='?')
    parser.add_argument('usage', nargs='?', choices=_VALID_USAGE, default='personal')
    parser.add_argument('--output', type=Path, help='Save complete JSON here instead of stdout')
    parser.add_argument('--brief-output', type=Path, help='Also save the full rendered brief here')
    parser.add_argument('--hf-username', help='Public Hugging Face username for likes and collections')
    parser.add_argument('--hf-import', action='append', default=[], help='Explicit Hugging Face model, dataset or Space URL')
    parser.add_argument('--refresh-hf-catalogue', action='store_true', help='Refresh the separate local Hugging Face catalogue')
    parser.add_argument('--diagnose', action='store_true', help='Print read-only configuration diagnostics and exit')
    args = parser.parse_args(argv)
    if args.diagnose:
        from oracle_diagnostic import diagnostic
        print(json.dumps(diagnostic(), indent=2))
        return 0
    if args.refresh_hf_catalogue and not args.query:
        if args.brief_output:
            parser.error('--brief-output requires a build query.')
        try:
            result = HuggingFaceProvider(username=args.hf_username).refresh_catalogue()
            text = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
            if args.output:
                _write_output(args.output, text)
                print(f'Saved Hugging Face catalogue result: {args.output}', file=sys.stderr)
            else:
                print(text, end='')
            return 0 if result.get('status') == 'available' else 1
        except Exception as exc:
            print(f'Hugging Face catalogue refresh failed ({type(exc).__name__}); previous complete catalogue was preserved.', file=sys.stderr)
            return 1
    if not args.query:
        parser.error('query is required unless --diagnose is used.')
    if args.output and args.brief_output and args.output.resolve() == args.brief_output.resolve():
        parser.error('JSON and brief outputs must use different paths.')
    try:
        result = relevancy_oracle({"query": args.query, "usage": args.usage,
                                   "hf_username": args.hf_username,
                                   "hf_imports": args.hf_import,
                                   "refresh_hf_catalogue": args.refresh_hf_catalogue})
        text = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
        if args.output:
            _write_output(args.output, text)
            print(f'Saved Oracle JSON: {args.output}', file=sys.stderr)
        else:
            print(text, end='')
        if args.brief_output:
            from render_brief import render_brief
            _write_output(args.brief_output, render_brief(result))
            print(f'Saved complete brief: {args.brief_output}', file=sys.stderr)
    except Exception as exc:
        print(f'Oracle failed ({type(exc).__name__}). Check access and provider status; existing saved JSON can be rendered separately.', file=sys.stderr)
        return 1
    return 0


def _write_output(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         delete=False, newline='\n') as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
