"""render_brief.py — human-readable Repo Scout brief over the oracle envelope.

``render_brief(response) -> str``. A deterministic renderer over the combined
JSON the relevancy oracle returns. No Claude call, no GitHub call: every
judgement it needs (relevance, what each candidate covers, licence and
maintenance flags) is already in the data, produced by the two halves. The
core oracle keeps ``brief`` null; this is the separate renderer the contract
calls for.

Output follows the v1.3 shape — one plain-language brief, then a compact
technical handoff: two outputs from one silent research run.

NOTE: built to the reconstructed v1.3 shape (silent research, no scored
tables, owned-vs-missing, brief + handoff). Check the section headings
against the canonical v1.3 system prompt and adjust to taste.

Output is kept ASCII-only so it prints cleanly in the Windows console.

CLI:
    python render_brief.py response.json
    python relevancy_oracle.py "I want to build X" personal | python render_brief.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

_USAGE = (
    "Usage: python render_brief.py response.json\n"
    '   or: python relevancy_oracle.py "..." personal | python render_brief.py'
)

# Inward matches are grouped by confidence, strongest first.
_CONFIDENCE_ORDER = ("high", "medium", "low")
_CONFIDENCE_LABEL = {
    "high": "Strong matches",
    "medium": "Possible matches",
    "low": "Loose connections",
}

_STALE_AFTER_DAYS = 14


def render_brief(response: dict) -> str:
    """Turn the combined oracle envelope into a plain-language brief string."""
    query = response.get("query") or "(no query)"
    inward = response.get("inward") or {}
    outward = response.get("outward") or {}
    snap = inward.get("index_snapshot") or {}
    matches = inward.get("matches") or []
    candidates = outward.get("candidates") or []
    gaps = outward.get("gaps_unfilled") or []
    huggingface = response.get("huggingface") or {}
    assembly = response.get("assembly") or {}

    lines: list[str] = []
    out = lines.append

    # --- Header ---------------------------------------------------------
    out(f"REPO SCOUT BRIEF -- {query}")
    meta = _meta_line(response, snap)
    if meta:
        out(meta)
    stale = _staleness_note(snap)
    if stale:
        out(stale)
    out("")

    # --- What you already have ------------------------------------------
    out("WHAT YOU ALREADY HAVE")
    if matches:
        out(_render_inward(matches))
    else:
        out("  Nothing in your index relates directly -- you're starting "
            "fresh on this one.")
    out("")

    # --- What you're missing --------------------------------------------
    out("WHAT YOU'RE MISSING")
    if candidates:
        out(_render_outward(candidates))
    else:
        out("  No external candidates surfaced -- either what you own covers "
            "it, or this needs a manual scout.")
    out("")

    reviews = outward.get('source_reviews') or candidates
    if response.get('source_inspection'):
        out('SOURCE INSPECTION AND REUSE')
        out('  Source was read at the revisions below. No downloaded code or integration tests were executed.')
        out('  Citations locate inspected text; engineering judgements still need the listed tests.')
        out('  Capability and absence judgements below apply only to inspected excerpts; absence across a whole repository is unverified.')
        out(_render_sources(matches + reviews))
        budget = response['source_inspection']
        out(f"  Inspection budget: {budget.get('repositories_attempted', 0)}/{budget.get('repository_limit', 0)} repositories attempted.")
        out("")

    if "huggingface" in response:
        out("HUGGING FACE RESOURCES")
        out(_render_huggingface(huggingface))
        out("")

    if "assembly" in response:
        out("ASSEMBLY ADVICE")
        out(_render_assembly(assembly))
        out("")

    # --- Gaps still unfilled --------------------------------------------
    if gaps:
        out("GAPS STILL UNFILLED")
        for gap in gaps:
            out(f"  - {gap}")
        out("")

    # --- Technical handoff ----------------------------------------------
    out("--- TECHNICAL HANDOFF ---")
    out(_render_handoff(matches, candidates, gaps))
    if response.get('source_inspection'):
        out('SOURCE REUSE HANDOFF:')
        for item in matches + reviews:
            assessment = item.get('source_assessment') or {}
            for component in assessment.get('components', []):
                out(f"  - {item.get('full_name')}@{assessment.get('revision')}: {component['name']} [{component['decision']}]")
                out('    ' + _reuse_files_line(component))
                out('    Validation: source inspected; execution and integration tests outstanding')
    if "huggingface" in response:
        out("HUGGING FACE HANDOFF:")
        for item in huggingface.get("assessments", []):
            out(f"  - {item.get('identity', '?')}@{item.get('revision', 'unresolved')}: "
                f"{item.get('status', 'unavailable')} [{item.get('evidence_quality', 'unavailable')}]")
            out("    Validation: card/configuration evidence inspected; weights and remote code not executed")
    if "assembly" in response:
        out("ASSEMBLY HANDOFF:")
        for selection in assembly.get("selections", []):
            out(f"  - {selection.get('requirement_id')}: {selection.get('resource_id')} "
                f"[{selection.get('decision')}]")
            out("    Tests outstanding: " + "; ".join(selection.get("tests", [])))

    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------
# Section renderers
# --------------------------------------------------------------------------

def _render_inward(matches: list[dict]) -> str:
    """Group owned repos by confidence, strongest first."""
    if any('source_assessment' in m for m in matches):
        return '\n'.join('  - ' + _source_summary(m, m.get('relevance') or m.get('summary', '')) for m in matches)
    buckets: dict[str, list[dict]] = {}
    for match in matches:
        conf = (match.get("confidence") or "low").lower()
        buckets.setdefault(conf, []).append(match)

    seen: set[str] = set()
    chunks: list[str] = []
    ordered = list(_CONFIDENCE_ORDER) + [
        c for c in buckets if c not in _CONFIDENCE_ORDER
    ]
    for conf in ordered:
        group = buckets.get(conf)
        if not group or conf in seen:
            continue
        seen.add(conf)
        chunks.append(f"  {_CONFIDENCE_LABEL.get(conf, conf.title())}:")
        for match in group:
            name = match.get("full_name", "?")
            detail = match.get("relevance") or match.get("summary") or "(no detail)"
            chunks.append(f"    - {name}: {detail}")
    return "\n".join(chunks)


def _render_outward(candidates: list[dict]) -> str:
    """Group missing candidates by layer, in first-seen order."""
    buckets: dict[str, list[dict]] = {}
    order: list[str] = []
    for cand in candidates:
        layer = cand.get("layer") or "other"
        if layer not in buckets:
            buckets[layer] = []
            order.append(layer)
        buckets[layer].append(cand)

    chunks: list[str] = []
    for layer in order:
        chunks.append(f"  {layer.title()} layer:")
        for cand in buckets[layer]:
            chunks.append("    - " + _candidate_line(cand))
    return "\n".join(chunks)


def _candidate_line(cand: dict) -> str:
    """One missing-repo recommendation as plain language."""
    name = cand.get("full_name", "?")
    parts = [_source_summary(cand, _fmt_covers(cand.get('covers'))) if 'source_assessment' in cand
             else f"{name}: covers {_fmt_covers(cand.get('covers'))}"]

    licence = cand.get("licence")
    if cand.get("usage_ok") is False:
        if licence and str(licence).lower() not in ("unverified", "unknown", ""):
            parts.append(f"licence {licence} unverified -- vet before use")
        else:
            parts.append("licence unverified -- vet before use")
    elif licence:
        parts.append(f"licence {licence}")

    maintenance = _maintenance_note(cand)
    if maintenance:
        parts.append(maintenance)

    line = "; ".join(parts)

    # `partial` may be a bool flag or a descriptive string explaining the
    # caveat — keep the explanation on its own line so the lead stays scannable.
    partial = cand.get("partial")
    if isinstance(partial, str) and partial.strip():
        line += f"\n        partial fit: {partial.strip()}"
    elif partial is True:
        line += "\n        partial fit"

    url = cand.get("url")
    if url:
        line += f"\n        {url}"
    return line


def _source_summary(item: dict, metadata: str) -> str:
    assessment = item.get('source_assessment') or {}
    components = assessment.get('components') or []
    useful = [c for c in components if c['decision'] not in ('do_not_use', 'uncertain')]
    prefix = str(item.get('full_name', '?')) + ': '
    if useful:
        return prefix + 'source-assessed possibility -- ' + '; '.join(c['purpose'] for c in useful)
    if assessment.get('status') == 'assessed':
        return prefix + 'source assessment recommends rejection or further investigation; see details below.'
    return prefix + f"[{assessment.get('status', 'not_inspected')}; metadata suggestion only] " + metadata


def _reuse_files_line(component: dict) -> str:
    decision = component['decision']
    if decision == 'do_not_use':
        return 'Reuse: rejected for this request.'
    paths = ', '.join(component['reuse_files'])
    if decision == 'uncertain':
        return 'Files needing further investigation: ' + (paths or 'No extraction boundary established')
    if decision in ('whole_project', 'library'):
        return 'Relevant files within the proposed project/library: ' + (paths or 'See source evidence')
    if decision == 'pattern':
        return 'Pattern reference files: ' + (paths or 'See source evidence')
    return 'Files to consider reusing after validation: ' + (paths or 'No extraction boundary established')


def _render_sources(items: list[dict]) -> str:
    lines, seen = [], set()
    for item in items:
        name = item.get('full_name', '?')
        if name in seen:
            continue
        seen.add(name)
        a = item.get('source_assessment') or {}
        lines.append(f"  {name}: {a.get('status', 'not_inspected')}")
        if a.get('revision'):
            lines.append(f"    Revision: {a['revision']}; inspected {a.get('inspected_at', 'incomplete')}")
        if a.get('files'):
            lines.append('    Files inspected: ' + ', '.join(f['path'] + (' [prefix only]' if f['truncated'] else '') for f in a['files']))
        for c in a.get('components', []):
            lines.extend([f"    {c['name']} -- {c['decision']}", f"      Helps with: {c['purpose']}",
                          f"      Reason: {c['rationale']}",
                          '      ' + _reuse_files_line(c),
                          f"      Coupling: {c['coupling']}"])
            for e in c['evidence']:
                lines.append(f"      Evidence: {e['url']}")
                lines.extend('        | ' + line for line in e['quote'].splitlines())
            for dep in c['dependencies']:
                lines.append(f"      Dependency [{dep['status']}]: {dep['name']} -- {dep['reason']}")
                for e in dep['evidence']:
                    lines.append(f"        {e['url']}")
            for dep in c.get('internal_dependencies', []):
                lines.append(f"      Internal import: {dep['path']} [{'inspected' if dep['inspected'] else 'not inspected'}]")
            for label, field in [('Adapt', 'changes_needed'), ('Test still to run', 'tests_to_run'), ('Unresolved', 'unknowns')]:
                lines.extend(f"      {label}: {value}" for value in c[field])
        licence = a.get('licence') or {}
        if licence:
            lines.append(f"    Licence evidence: {licence['status']}; " + ', '.join(licence.get('paths', [])))
            lines.append('      ' + licence.get('notes', 'Reuse conditions remain unresolved.'))
        lines.extend('    Limitation: ' + x for x in a.get('limitations', []))
    return '\n'.join(lines)


def _render_huggingface(block: dict) -> str:
    lines = [f"  Provider status: {block.get('status', 'unavailable')}"]
    lines.append(f"  Query planner: {block.get('query_planner', 'unavailable')}")
    for item in block.get("query_plan", []):
        query = item.get("query") or "(filter only)"
        task_filter = item.get("task_filter") or "none"
        lines.append(f"  Search capability: {item.get('capability', 'unspecified')}; "
                     f"query={query}; task filter={task_filter}; "
                     f"types={','.join(item.get('repo_types', []))}")
    catalogue = block.get("catalogue") or {}
    lines.append(f"  Personal catalogue: {catalogue.get('status', 'unconfigured')}"
                 + (f" ({catalogue.get('username')})" if catalogue.get('username') else ""))
    searches = block.get("public_search") or {}
    for repo_type in ("model", "dataset", "space"):
        result = searches.get(repo_type) or {}
        lines.append(f"  Public {repo_type}s: {result.get('status', 'unavailable')}; "
                     f"{len(result.get('items', []))} result(s) retained")
    assessments = block.get("assessments") or []
    if not assessments:
        lines.append("  No Hugging Face resource reached evidence inspection.")
    for item in assessments:
        lines.append(f"  - {item.get('identity', '?')}: {item.get('status', 'unavailable')}")
        if item.get("revision"):
            lines.append(f"    Revision: {item['revision']}")
        if item.get("task"):
            lines.append(f"    Task: {item['task']}")
        if item.get("library"):
            lines.append(f"    Library: {item['library']}")
        licence = item.get("licence") or {}
        lines.append(f"    Licence evidence: {licence.get('value', 'unverified')} "
                     f"[{licence.get('evidence', 'none')}]; usage_ok={licence.get('usage_ok', False)}")
        lines.append(f"    Evidence quality: {item.get('evidence_quality', 'unavailable')}; "
                     f"eligible_for_assembly={item.get('eligible_for_assembly', False)}")
        for evidence in item.get("evidence", []):
            lines.append(f"    Evidence [{evidence.get('kind')}]: {evidence.get('url')}")
            if evidence.get("truncated"):
                lines.append("      Prefix only; conclusions remain limited.")
        for requirement in item.get("deployment_requirements", []):
            lines.append(f"    Deployment evidence: {requirement.get('quote')} ({requirement.get('url')})")
        fit = item.get("fit_assessment") or {}
        if fit:
            lines.append(f"    Fit decision: {fit.get('decision', 'uncertain')} -- {fit.get('purpose', '')}")
            lines.append(f"      Reason: {fit.get('rationale', '')}")
            lines.append(f"      Coupling: {fit.get('coupling', 'unresolved')}")
            for citation in fit.get("evidence", []):
                lines.append(f"      Fit evidence [{citation.get('kind')}]: {citation.get('url')}")
                lines.extend("        | " + line for line in citation.get("quote", "").splitlines())
            for dependency in fit.get("dependencies", []):
                lines.append(f"      Dependency [{dependency.get('status')}]: {dependency.get('name')} -- {dependency.get('reason')}")
            for label, key in (("Interface", "interfaces"), ("Adapt", "adaptations"),
                               ("Test still to run", "tests"), ("Assumption", "assumptions"),
                               ("Unresolved", "unknowns")):
                lines.extend(f"      {label}: {value}" for value in fit.get(key, []))
        lines.extend("    Limitation: " + value for value in item.get("limitations", []))
    lines.extend("  Provider limitation: " + value for value in block.get("limitations", []))
    return "\n".join(lines)


def _render_assembly(block: dict) -> str:
    lines = [f"  Status: {block.get('status', 'incomplete')}; execution=not_run; coverage_complete=false"]
    requirements = block.get("requirements") or []
    if requirements:
        lines.append("  Requirements:")
        for requirement in requirements:
            lines.append(f"    - {requirement.get('id')}: {requirement.get('need')} "
                         f"[{requirement.get('status', 'open')}]")
            lines.extend(f"      Acceptance: {value}" for value in requirement.get("acceptance", []))
    selections = block.get("selections") or []
    if selections:
        lines.append("  Proposed minimal assembly:")
    for selection in selections:
        lines.append(f"    - {selection.get('resource_id')} / {selection.get('component_name')} "
                     f"[{selection.get('decision')}] -> {selection.get('requirement_id')}")
        lines.append(f"      Role: {selection.get('role')}")
        lines.append(f"      Why this scope: {selection.get('rationale')}")
        for label, key in (("Interface", "interfaces"), ("Connect", "connections"),
                           ("Adapt", "adaptations"), ("Conflict", "conflicts"),
                           ("Assumption", "assumptions"), ("Test", "tests")):
            lines.extend(f"      {label}: {value}" for value in selection.get(key, []))
    if block.get("data_flow"):
        lines.append("  Data flow:")
        lines.extend(f"    {index}. {value}" for index, value in enumerate(block["data_flow"], 1))
    if block.get("open_requirements"):
        lines.append("  Open requirements:")
        lines.extend("    - " + value for value in block["open_requirements"])
    if block.get("global_tests"):
        lines.append("  End-to-end validation still required:")
        lines.extend("    - " + value for value in block["global_tests"])
    lines.extend("  Limitation: " + value for value in block.get("limitations", []))
    return "\n".join(lines)


def _fmt_covers(covers) -> str:
    """`covers` may arrive as a sentence (string) or a list of scopes."""
    if isinstance(covers, str):
        return covers.strip() or "unspecified scope"
    if isinstance(covers, (list, tuple)):
        items = [str(c).strip() for c in covers if str(c).strip()]
        return ", ".join(items) if items else "unspecified scope"
    return "unspecified scope"


def _maintenance_note(cand: dict) -> str:
    last = cand.get("last_push")
    last_date = str(last)[:10] if last else ""  # trim full ISO timestamp to date
    if cand.get("maintained") is False:
        return "appears unmaintained" + (
            f" (last push {last_date})" if last_date else ""
        )
    if last_date:
        return f"last push {last_date}"
    return ""


def _render_handoff(
    matches: list[dict], candidates: list[dict], gaps: list[str]
) -> str:
    """Compact, scan-friendly summary for the build step."""
    lines: list[str] = []

    owned = ", ".join(m.get("full_name", "?") for m in matches) or "(none)"
    lines.append(f"OWNED: {owned}")

    if candidates:
        lines.append("FETCH:")
        for cand in candidates:
            flag = "" if cand.get("usage_ok") is not False else " [licence unverified]"
            layer = cand.get("layer", "other")
            url = cand.get("url", "")
            lines.append(
                f"  - {cand.get('full_name', '?')} [{layer}]{flag} {url}".rstrip()
            )
    else:
        lines.append("FETCH: (none)")

    lines.append("OPEN: " + ("; ".join(gaps) if gaps else "(none)"))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Header helpers
# --------------------------------------------------------------------------

def _meta_line(response: dict, snap: dict) -> str:
    bits: list[str] = []
    gen = response.get("generated_at")
    if gen:
        bits.append(f"generated {gen[:10]}")
    ver = response.get("oracle_version")
    if ver:
        bits.append(f"oracle v{ver}")
    count = snap.get("count")
    if count is not None:
        regen = snap.get("generated_at")
        regen_txt = f", regenerated {regen[:10]}" if regen else ""
        bits.append(f"index: {count} repos{regen_txt}")
    return "  " + " | ".join(bits) if bits else ""


def _staleness_note(snap: dict) -> str:
    """Warn if the index hasn't been regenerated recently."""
    raw = snap.get("generated_at")
    if not raw:
        return ""
    try:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return ""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - ts).days
    if age_days > _STALE_AFTER_DAYS:
        return (
            f"  Note: index last regenerated {age_days} days ago -- repos "
            "starred since then won't appear here."
        )
    return ""


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _cli(argv: list[str]) -> int:
    if argv[:1] in (["-h"], ["--help"]):
        print(_USAGE)
        return 0

    if argv:
        try:
            with open(argv[0], "r", encoding="utf-8") as handle:
                raw = handle.read()
        except OSError as exc:
            print(f"Error: cannot read {argv[0]} ({exc})", file=sys.stderr)
            return 2
    else:
        raw = sys.stdin.read()

    if not raw.strip():
        print(_USAGE, file=sys.stderr)
        return 2

    try:
        response, skipped = _parse_envelope(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"Error: input is not valid JSON ({exc})", file=sys.stderr)
        return 2

    if skipped:
        print(
            f"Warning: skipped {skipped} bytes of non-JSON before the "
            "envelope. The oracle is writing progress to stdout -- route those "
            "prints to stderr so stdout stays JSON-clean (the contract requires "
            "it, and the SBOS watcher will need it too).",
            file=sys.stderr,
        )

    print(render_brief(response))
    return 0


def _parse_envelope(raw: str) -> tuple[dict, int]:
    """Parse the envelope, tolerating leading non-JSON noise.

    Returns ``(response, skipped)`` where ``skipped`` is the count of leading
    bytes that were not part of the JSON — 0 when the input was already clean.
    Tolerance is a convenience; the proper fix is keeping stdout JSON-clean.
    """
    try:
        return json.loads(raw), 0
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    if start == -1:
        raise ValueError("no JSON object found in input")
    obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    return obj, start


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
