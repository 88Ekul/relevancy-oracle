"""
health_check.py — maintenance enrichment pass for the Stars Indexer catalogue.

Reads stars_index.json and, for every repo, asks the GitHub API for its real
maintenance signals (last push date, archived/disabled status). It writes those
facts back into the catalogue with a fetched-at timestamp and a plain health
label, then prints a short report of which owned repos look stale or abandoned.

Why this is a separate pass from indexer.py:
  - Summarising a README (indexer.py, via Claude) is expensive and rarely needs
    redoing.
  - Checking whether a repo is still maintained is cheap and goes stale fast.
  Splitting them lets you refresh health often and summaries seldom. It also
  sidesteps stars_progress.json's skip-already-indexed logic, which otherwise
  means existing entries never get re-examined for maintenance.

On the statelessness principle: the relevancy oracle never persists AI
judgements, to avoid compounding errors. This is different — pushed_at and
archived are OBJECTIVE facts from GitHub, stored alongside a health_checked_at
timestamp so their age is always visible. That is the provenance-and-expiry
pattern the contract endorses, not a cached opinion.

British English throughout. GitHub's API spells its fields in American English
(pushed_at, archived); those are kept as received on their own surface, and our
derived keys (health, health_checked_at) use British-neutral names.

Run:
    python health_check.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────
INDEX_PATH = Path(os.getenv("STARS_INDEX_PATH", "stars_index.json"))
GITHUB_API = "https://api.github.com"
STALE_AFTER_DAYS = 365          # the contract's 12-month maintenance line
REQUEST_PAUSE = 0.1             # small courtesy pause between calls


def _headers() -> dict:
    token = os.getenv("GITHUB_TOKEN")
    if not token or token == "your_token_here":
        print("Error: GITHUB_TOKEN not set. Add it to your .env file.")
        raise SystemExit(1)
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }


# ── Catalogue loading (preserves the on-disk shape for write-back) ─────
def _load_raw():
    """Load the index and return (raw_object, list_of_entry_dicts).

    The raw object is whatever shape the file is in (list, wrapper dict, or
    full_name -> entry map). The entry list holds references INTO that raw
    object, so mutating an entry in place updates what we later write back.
    """
    if not INDEX_PATH.exists():
        print(f"Error: catalogue not found at {INDEX_PATH}.")
        print("Set STARS_INDEX_PATH in .env, or run the indexer first.")
        raise SystemExit(1)

    raw = json.loads(INDEX_PATH.read_text(encoding="utf-8"))

    if isinstance(raw, list):
        entries = [e for e in raw if isinstance(e, dict)]
    elif isinstance(raw, dict):
        listish = next(
            (raw[k] for k in ("repos", "index", "items", "stars")
             if isinstance(raw.get(k), list)),
            None,
        )
        if listish is not None:
            entries = [e for e in listish if isinstance(e, dict)]
        else:
            entries = []
            for name, entry in raw.items():
                if isinstance(entry, dict):
                    entry.setdefault("full_name", name)
                    entries.append(entry)
    else:
        print("Error: unexpected stars_index.json shape.")
        raise SystemExit(1)

    return raw, entries


def _owner_repo(entry: dict):
    """Pull owner/repo out of an entry, tolerant of key naming."""
    name = entry.get("full_name") or entry.get("name") or entry.get("repo")
    if not name or "/" not in name:
        return None
    owner, repo = name.split("/", 1)
    return owner.strip(), repo.strip()


def _classify(pushed_at: str, archived: bool, disabled: bool) -> str:
    """Return a plain health label from the raw GitHub signals."""
    if archived or disabled:
        return "archived"
    if not pushed_at:
        return "unknown"
    try:
        pushed = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
    except ValueError:
        return "unknown"
    age_days = (datetime.now(timezone.utc) - pushed).days
    return "maintained" if age_days <= STALE_AFTER_DAYS else "stale"


def _atomic_write(raw_object) -> None:
    """Write the enriched index back safely (temp file, then replace)."""
    tmp = INDEX_PATH.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(raw_object, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tmp.replace(INDEX_PATH)


# ── Main ───────────────────────────────────────────────────────────────
def main() -> int:
    raw, entries = _load_raw()
    total = len(entries)
    if total == 0:
        print("No repositories found in the index.")
        return 0

    print(f"Health-checking {total} repositories…\n")
    headers = _headers()
    checked_at = datetime.now(timezone.utc).isoformat()

    tally = {"maintained": 0, "stale": 0, "archived": 0, "unavailable": 0, "unknown": 0}
    flagged: list[tuple[str, str, str]] = []   # (full_name, health, pushed_at)

    for i, entry in enumerate(entries, 1):
        parsed = _owner_repo(entry)
        if parsed is None:
            entry["health"] = "unknown"
            entry["health_checked_at"] = checked_at
            tally["unknown"] += 1
            continue

        owner, repo = parsed
        full_name = f"{owner}/{repo}"
        try:
            resp = requests.get(
                f"{GITHUB_API}/repos/{owner}/{repo}", headers=headers, timeout=20
            )
        except requests.RequestException as exc:
            print(f"  [{i}/{total}] {full_name} — request failed: {exc}")
            entry["health"] = "unavailable"
            entry["health_checked_at"] = checked_at
            tally["unavailable"] += 1
            continue

        if resp.status_code == 404:
            # Repo deleted, renamed, or made private since starring.
            entry["health"] = "unavailable"
            entry["health_checked_at"] = checked_at
            tally["unavailable"] += 1
            flagged.append((full_name, "unavailable", "—"))
            print(f"  [{i}/{total}] {full_name} — not found (deleted or private)")
            continue

        if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
            reset = resp.headers.get("X-RateLimit-Reset", "?")
            print(f"\nRate limit reached after {i - 1} repos. Resets at epoch {reset}.")
            print("Saving progress so far; re-run later to finish the rest.")
            break

        if resp.status_code != 200:
            print(f"  [{i}/{total}] {full_name} — HTTP {resp.status_code}")
            entry["health"] = "unavailable"
            entry["health_checked_at"] = checked_at
            tally["unavailable"] += 1
            continue

        data = resp.json()
        pushed_at = data.get("pushed_at", "")
        archived = bool(data.get("archived"))
        disabled = bool(data.get("disabled"))
        health = _classify(pushed_at, archived, disabled)

        entry["pushed_at"] = pushed_at
        entry["archived"] = archived
        entry["health"] = health
        entry["health_checked_at"] = checked_at

        tally[health] = tally.get(health, 0) + 1
        if health in ("stale", "archived"):
            flagged.append((full_name, health, pushed_at[:10] if pushed_at else "—"))

        time.sleep(REQUEST_PAUSE)

    _atomic_write(raw)

    # ── Report ─────────────────────────────────────────────────────────
    print("\nDone. Health written back to the index.\n")
    print("  Maintained (pushed within 12 months) :", tally["maintained"])
    print("  Stale (no push in over 12 months)     :", tally["stale"])
    print("  Archived (explicitly archived)        :", tally["archived"])
    print("  Unavailable (deleted/private/error)   :", tally["unavailable"])
    if tally["unknown"]:
        print("  Unknown (no usable repo name)         :", tally["unknown"])

    if flagged:
        print("\nWorth a look — these owned repos may be unmaintained:")
        for full_name, health, when in sorted(flagged, key=lambda x: x[1]):
            label = "archived" if health == "archived" else (
                "unavailable" if health == "unavailable" else f"last push {when}")
            print(f"  - {full_name}  ({label})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
