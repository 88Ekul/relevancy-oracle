"""relevancy_oracle.py — the combined relevancy oracle.

A thin orchestrator that runs both halves of the oracle and returns one
combined answer to: "I want to build X — what do I already have, and
what do I need?"

    inward_check  -> what you already own (starred repos that match)
    outward_check -> what you're missing (vetted candidates for the gaps)

The integration that makes the combined run smarter than either half
alone: inward runs first, and its matches are handed to outward so
outward excludes repos you already own and recommends only genuine gaps.

JSON-clean in, JSON-clean out. Private run journals preserve completed work
and spending; catalogues are not enriched with Oracle judgements. Imports
nothing from Second-Brain-OS — the dependency runs one way only.

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
import re
from pathlib import Path
from datetime import datetime, timezone

from inward_check import inward_check, _load_catalogue, _compact, INDEX_PATH
from outward_check import outward_check
from assembly_advice import AssemblyAdviser
from huggingface_provider import HuggingFaceProvider, HuggingFaceClient, HFLimits
from source_inspection import SourceInspector
from oracle_costs import RunSession, MODES, CostControlError, BudgetExceeded, atomic_json

ORACLE_VERSION = "0.4"

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
                     assembler: AssemblyAdviser | None = None, *,
                     mode="standard", max_cost_usd=1.0, model_profile="quality",
                     run_dir=None, resume=False, cache_dir=None) -> dict:
    """Run with durable checkpoints and a total budget, including prior attempts."""
    req = _normalise_request(request)
    hf_path = Path(os.getenv('HF_CATALOGUE_PATH', 'hf_catalogue.json'))
    with RunSession(req, mode=mode, max_cost_usd=max_cost_usd, model_profile=model_profile,
                    run_dir=run_dir, resume=resume, cache_dir=cache_dir,
                    catalogue_paths=(INDEX_PATH,) if req['refresh_hf_catalogue'] else (INDEX_PATH, hf_path)) as session:
        progress = {"query": req["query"], "oracle_version": ORACLE_VERSION,
                    "generated_at": datetime.now(timezone.utc).isoformat(), "brief": None,
                    "inward": {"matches": []}, "outward": {"candidates": [], "gaps_unfilled": []}}
        try:
            if mode == 'quick':
                result = session.stage('quick', lambda: _quick(req))
            else:
                limits = MODES[mode]
                inspector = inspector or SourceInspector(max_repositories=limits['github_repositories'],
                                                          inward_limit=limits['inward_repositories'])
                hf_provider = hf_provider or HuggingFaceProvider(
                    HuggingFaceClient(limits=HFLimits(max_assessments=limits['hf_assessments'])),
                    username=req.get('hf_username'))
                result = _run_engine(req, inspector, hf_provider, assembler, session, progress)
            result['run_status'] = 'complete'
        except CostControlError as exc:
            result = progress
            result['run_status'] = 'budget_stopped' if isinstance(exc, BudgetExceeded) else 'stopped'
            result['stop_reason'] = str(exc)
            result['outward'].setdefault('gaps_unfilled', []).append(
                'Assessment stopped before completion: ' + str(exc))
            if inspector is not None:
                result['source_inspection'] = inspector.report()
        session.data['status'] = result['run_status']
        session.save()
        result['cost_control'] = session.report()
        if 'source_inspection' in result:
            # The journal includes earlier attempts on resume; inspector object
            # counters only cover work executed by this process.
            source_usage = [v for k, v in result['cost_control']['stages'].items()
                            if k in ('source_select', 'source_analysis')]
            result['source_inspection'].update(
                model_calls=sum(v['calls'] for v in source_usage),
                input_tokens=sum(v['input_tokens'] for v in source_usage),
                output_tokens=sum(v['output_tokens'] for v in source_usage))
        atomic_json(session.path / 'response.json', result)
        return result


def _quick(req):
    """Free local keyword shortlist; makes no model or external provider calls."""
    entries, snapshot = _load_catalogue()
    stopwords = {'the', 'and', 'for', 'with', 'what', 'have', 'already', 'want', 'build',
                 'need', 'using', 'from', 'that', 'this', 'missing'}
    terms = set(re.findall(r'[a-z0-9]+', req['query'].lower())) - stopwords
    terms = {t for t in terms if len(t) > 2}
    def score(entry):
        words = set(re.findall(r'[a-z0-9]+', _compact(entry).lower()))
        return len(terms & words)
    ranked = sorted(enumerate(entries), key=lambda pair: (-score(pair[1]), pair[0]))
    matches = [{"full_name": e.get('full_name', e.get('name', '?')),
                "summary": e.get('summary', ''), "confidence": 'low',
                "relevance": 'Keyword overlap only; relevance, source and licence have not been assessed.'}
               for _, e in ranked if score(e) > 0][:12]
    hf_path = Path(os.getenv('HF_CATALOGUE_PATH', 'hf_catalogue.json'))
    hf_matches = []
    if hf_path.is_file():
        try:
            catalogue = json.loads(hf_path.read_text(encoding='utf-8'))
            hf_matches = [item for item in catalogue.get('resources', []) if score(item) > 0][:12]
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    return {'query': req['query'], 'oracle_version': ORACLE_VERSION,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'inward': {'index_snapshot': snapshot, 'matches': matches},
            'outward': {'candidates': [], 'gaps_unfilled': [
                'Quick mode is a local keyword shortlist only. No live outward search, source inspection, licence assessment or assembly was performed. Non-obvious matches may be missed.']},
            'quick_huggingface_matches': hf_matches, 'brief': None}


def _run_engine(request, inspector, hf_provider, assembler, session, progress):
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
    inward = session.stage('inward', lambda: inward_check(req))
    progress['inward'] = inward

    # 2. Outward second — hand it inward's matches so it excludes repos
    #    already owned and recommends only the genuine gaps. This is the
    #    integration: the combined run beats either half in isolation.
    inward_matches = inward.get("matches", []) if isinstance(inward, dict) else []
    def inspect_owned():
        inspector.inspect_inward(inward_matches, req)
        return {'matches': inward_matches, 'results': getattr(inspector, 'results', {}),
                'report': inspector.report()}
    owned = session.stage('inward_source', inspect_owned)
    inward_matches = owned['matches']
    inward['matches'] = inward_matches
    if hasattr(inspector, 'results'):
        inspector.results = owned['results']
    def check_outward():
        result = outward_check(req, inward_matches=inward_matches, inspector=inspector)
        return {'outward': result, 'results': getattr(inspector, 'results', {}),
                'report': inspector.report()}
    external = session.stage('outward', check_outward)
    outward = external['outward']
    progress['outward'] = outward
    if hasattr(inspector, 'results'):
        inspector.results = external['results']
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
        huggingface = session.stage('huggingface', lambda: hf_provider.collect(hf_request))
    except CostControlError:
        raise
    except Exception as exc:
        huggingface = {'status': 'unavailable', 'provider': 'huggingface',
                       'catalogue': {'status': 'unavailable', 'resources': []},
                       'public_search': {}, 'imports': [], 'assessments': [],
                       'limitations': [f'Hugging Face provider failed ({type(exc).__name__}); resources remain unresolved.'],
                       'execution': 'not_run'}
    progress['huggingface'] = huggingface
    if huggingface.get('status') == 'unavailable':
        outward.setdefault('gaps_unfilled', []).append(
            'Hugging Face discovery was unavailable; model, dataset and Space options remain unresolved.')
    elif huggingface.get('status') == 'no_matches':
        outward.setdefault('gaps_unfilled', []).append(
            'Hugging Face searches completed but returned no matches for the planned ML capabilities.')

    # 4. Cross-source advice may select only validated inspected resources.
    outward_items = outward.get('source_reviews') or outward.get('candidates') or []
    assembly = session.stage('assembly', lambda: (assembler or AssemblyAdviser()).assemble(
        req, inward_matches, outward_items, huggingface, outward.get('identified_gaps', [])))

    # 5. Assemble the contract envelope. `brief` stays null by design.
    return {
        "query": req["query"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "oracle_version": ORACLE_VERSION,
        "inward": inward,
        "outward": outward,
        "source_inspection": external['report'],
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
    parser.add_argument('--mode', choices=tuple(MODES), default='standard',
                        help='quick: free local keywords; standard: bounded source advice; deep: wider inspection')
    parser.add_argument('--max-cost-usd', type=float, default=1.0,
                        help='Total run budget, including previous attempts on resume (default: 1 USD)')
    parser.add_argument('--model-profile', choices=('quality', 'economical'), default='quality',
                        help='Economical uses cheaper planning/vetting models; quality comparison is still pending')
    parser.add_argument('--run-dir', type=Path, help='Directory for private usage journal and checkpoints')
    parser.add_argument('--resume', action='store_true', help='Resume --run-dir with unchanged request and settings')
    parser.add_argument('--estimate', action='store_true', help='Show offline mode limits and budget, without model or network calls')
    args = parser.parse_args(argv)
    if args.resume and not args.run_dir:
        parser.error('--resume requires --run-dir.')
    if args.estimate:
        print(json.dumps({'mode': args.mode, 'limits': MODES[args.mode],
                          'budget_usd': args.max_cost_usd, 'model_profile': args.model_profile,
                          'estimated_cost_usd': 0 if args.mode == 'quick' else None,
                          'note': 'Quick is free. Other costs depend on retrieved evidence; each paid call is token-counted and reserved before sending. A budget is not a completion quote.'}, indent=2))
        return 0
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
                                   "refresh_hf_catalogue": args.refresh_hf_catalogue},
                                  mode=args.mode, max_cost_usd=args.max_cost_usd,
                                  model_profile=args.model_profile, run_dir=args.run_dir,
                                  resume=args.resume)
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
    return 0 if result.get('run_status', 'complete') == 'complete' else 3


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
