"""
inward_check — the inward half of the relevancy oracle.

Reads the local Stars Indexer catalogue (stars_index.json) and reasons over it
with Claude to surface which repositories the user ALREADY OWNS that relate to a
given build idea. Returns the `inward` block of the relevancy-oracle response
envelope defined in INTERFACE_CONTRACT.md (v0.1).

Contract clauses honoured here:
  - One-way dependency: this module imports nothing from Second-Brain-OS.
  - JSON-clean: input and output are plain, serialisable dicts, so a later
    drop-file watcher can wrap this with zero change to the core.
  - Stateless: the catalogue is read fresh and reasoned over on every call;
    relevance judgements are returned to the caller and NEVER written back into
    stars_index.json (the compounding-errors mitigation).
  - British English throughout, including output keys. The inward half never
    touches the GitHub API, so no licence/license spelling boundary applies here.

Standalone use:
    python inward_check.py "I want to build a WhatsApp CRM for tradespeople"
"""
from __future__ import annotations

import json
import os
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

from anthropic import Anthropic
from oracle_costs import message_call, RunSession, CostControlError
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────
# Everything overridable via .env so nothing that might rotate is hardcoded.
INDEX_PATH = Path(os.getenv("STARS_INDEX_PATH", "stars_index.json"))
MODEL = os.getenv("ORACLE_MODEL", "claude-opus-4-5")
MAX_ENTRY_CHARS = 1200   # cap per-repo text shipped into the prompt
MAX_MATCHES = 12         # most matches returned in one call

CONTRACT_VERSION = "0.1"


# ── Catalogue loading ─────────────────────────────────────────────────
def _load_catalogue() -> tuple[list[dict], dict]:
    """Load stars_index.json fresh on every call. Returns (entries, snapshot).

    Tolerant of three on-disk shapes: a bare list of entries, a wrapper dict
    such as {"generated_at": ..., "repos": [...]}, or a map of
    full_name -> entry. The snapshot records count and generation time so any
    caller can see whether it reasoned over a stale index.
    """
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Catalogue not found at {INDEX_PATH}. "
            "Set STARS_INDEX_PATH in .env, or run the indexer first."
        )

    raw = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    generated_at = None

    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict):
        generated_at = (
            raw.get("generated_at") or raw.get("updated_at") or raw.get("updated")
        )
        listish = next(
            (raw[k] for k in ("repos", "index", "items", "stars")
             if isinstance(raw.get(k), list)),
            None,
        )
        if listish is not None:
            entries = listish
        else:
            # Treat the dict as a full_name -> entry map.
            entries = []
            for name, entry in raw.items():
                if isinstance(entry, dict):
                    entry.setdefault("full_name", name)
                    entries.append(entry)
    else:
        raise ValueError("Unexpected stars_index.json shape.")

    if generated_at is None:
        # Honest provenance fallback: the file's own modification time.
        mtime = datetime.fromtimestamp(INDEX_PATH.stat().st_mtime, tz=timezone.utc)
        generated_at = mtime.isoformat()

    return entries, {"count": len(entries), "generated_at": generated_at}


def _compact(entry: dict) -> str:
    """Trim one catalogue entry to a compact block for the prompt.

    Field-name agnostic on purpose: the model reads whatever survives, so a
    slightly different schema still works. Raw README bulk is dropped so it
    never bloats the prompt.
    """
    name = (
        entry.get("full_name")
        or entry.get("name")
        or entry.get("repo")
        or "unknown/unknown"
    )
    keep = {k: v for k, v in entry.items()
            if k not in ("readme", "raw_readme", "readme_text")}
    blob = json.dumps(keep, ensure_ascii=False)
    if len(blob) > MAX_ENTRY_CHARS:
        blob = blob[:MAX_ENTRY_CHARS] + "…"
    return f"{name}: {blob}"


# ── Reasoning ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are the inward check of a personal repository relevancy oracle.

You are given a build idea and a catalogue of GitHub repositories the user has
already starred (their existing arsenal). Your job is to surface which of these
OWNED repositories relate to the idea — including non-obvious, imaginative
connections, not just literal keyword matches. The whole point is to remind the
user of possibilities hiding in things they already own.

Judge relevance on what each repo could contribute to THIS idea: a reusable
pattern, a single component, a whole solution, or an adjacent capability. Prefer
a smaller set of genuinely useful matches over a long list of weak ones. If
nothing relates, return an empty list — never invent a match to fill space.

Reason fresh from the catalogue every time. Do not assume any prior judgement.

Return ONLY valid JSON — no preamble, no Markdown fences — in this exact shape:
{
  "matches": [
    {
      "full_name": "owner/repo",
      "category": "the repo's category from the catalogue",
      "summary": "the repo's existing summary from the catalogue, unchanged",
      "relevance": "one or two plain sentences: why this owned repo relates to THIS idea",
      "confidence": "high | medium | low"
    }
  ]
}

Write all prose in British English (summarise, colour, organisation, behaviour).
Echo full_name, category and summary from the catalogue unchanged; generate
relevance and confidence fresh. Return at most %%MAX_MATCHES%% matches, best first.
""".strip()


def _parse_matches(response) -> list[dict]:
    """Pull the matches list out of the model response, defensively."""
    text = "".join(
        b.text for b in response.content if getattr(b, "type", None) == "text"
    ).strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return []
        data = json.loads(text[start:end + 1])

    raw_matches = data.get("matches", []) if isinstance(data, dict) else []
    clean: list[dict] = []
    for m in raw_matches:
        if isinstance(m, dict) and m.get("full_name"):
            clean.append({
                "full_name": m.get("full_name"),
                "category": m.get("category", "Other"),
                "summary": m.get("summary", ""),
                "relevance": m.get("relevance", ""),
                "confidence": m.get("confidence", "medium"),
            })
    return clean


def inward_check(request: dict) -> dict:
    """Run the inward check for a build idea.

    Args:
        request: a dict with at least {"query": str}. "audience" is read if
            present to colour the matching; all other oracle fields are ignored
            by the inward half.

    Returns:
        The `inward` block of the relevancy-oracle envelope:
        {"index_snapshot": {"count": int, "generated_at": str}, "matches": [...]}.
    """
    query = (request or {}).get("query")
    if not query or not str(query).strip():
        raise ValueError("request['query'] is required and must be non-empty.")
    audience = (request or {}).get("audience")

    entries, snapshot = _load_catalogue()
    catalogue = "\n".join(_compact(e) for e in entries)

    user_block = f"BUILD IDEA:\n{query}\n"
    if audience:
        user_block += f"\nWHO IT IS FOR:\n{audience}\n"
    user_block += (
        f"\nCATALOGUE ({snapshot['count']} owned repositories):\n{catalogue}"
    )

    client = Anthropic(timeout=180.0, max_retries=0)
    system = SYSTEM_PROMPT.replace("%%MAX_MATCHES%%", str(MAX_MATCHES))

    response = message_call(client, 'inward',
        model=MODEL,
        max_tokens=4000,
        system=system,
        messages=[{"role": "user", "content": user_block}],
    )

    matches = _parse_matches(response)
    # Stateless: nothing is written back to the catalogue. The judgements above
    # exist only in this return value.
    return {"index_snapshot": snapshot, "matches": matches[:MAX_MATCHES]}


# ── Thin standalone CLI (the same core Layer 4.5 will import) ──────────
def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description='Budgeted catalogue reasoning only.')
    parser.add_argument('query')
    parser.add_argument('--max-cost-usd', type=float, default=1.0)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv[1:])
    if args.resume and not args.run_dir:
        parser.error('--resume requires --run-dir.')
    request = {'query': args.query, 'workflow': 'inward_only'}
    try:
        with RunSession(request, max_cost_usd=args.max_cost_usd, run_dir=args.run_dir,
                        resume=args.resume, catalogue_paths=[INDEX_PATH]) as session:
            result = session.stage('inward', lambda: inward_check(request))
            result['cost_control'] = session.report()
            session.data['status'] = 'complete'
            session.save()
    except CostControlError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
