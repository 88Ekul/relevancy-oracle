#!/usr/bin/env python3
"""
GitHub Stars Indexer
Pulls all starred repos, fetches READMEs, generates AI summaries + use cases.
Output: stars_index.json
"""

import os
import json
import time
import requests
import anthropic
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OUTPUT_FILE = "stars_index.json"
PROGRESS_FILE = "stars_progress.json"  # Resume support
RATE_LIMIT_PAUSE = 1.2  # seconds between GitHub requests
BATCH_PAUSE = 3         # seconds between Anthropic calls

# ── GitHub helpers ─────────────────────────────────────────────────────────────
GH_HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3.star+json"
}

def get_all_stars():
    """Fetch all starred repos with pagination."""
    stars = []
    page = 1
    print("Fetching starred repos from GitHub...")
    while True:
        resp = requests.get(
            "https://api.github.com/user/starred",
            headers=GH_HEADERS,
            params={"per_page": 100, "page": page}
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        stars.extend(data)
        print(f"  Page {page}: {len(data)} repos (total so far: {len(stars)})")
        page += 1
        time.sleep(RATE_LIMIT_PAUSE)
    print(f"Total starred repos found: {len(stars)}\n")
    return stars

def get_readme(owner, repo):
    """Fetch README content, return plain text (truncated to ~3000 chars)."""
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/readme",
            headers={**GH_HEADERS, "Accept": "application/vnd.github.v3.raw"}
        )
        if resp.status_code == 200:
            text = resp.text[:3000]
            return text
    except Exception:
        pass
    return ""

# ── Anthropic helper ───────────────────────────────────────────────────────────
# Locked taxonomy — must match the list in the summarise_repo prompt exactly.
# Any value the model returns that isn't in this set is corrected to "Other"
# so drift (e.g. "DevOps-Infrastructure") never silently enters the index.
VALID_CATEGORIES = {
    "CLI-Tool", "Library", "Framework", "Boilerplate", "AI-ML",
    "Automation", "Data", "DevOps", "Frontend", "Backend", "Full-Stack",
    "Learning-Resource", "API-Integration", "Security", "Other",
}

def _normalise_category(raw: str) -> str:
    """Return the category unchanged if it is valid, else fall back to Other."""
    if raw in VALID_CATEGORIES:
        return raw
    corrected = "Other"
    print(f"    ⚠ Unknown category '{raw}' — corrected to '{corrected}'")
    return corrected

client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

SYSTEM_PROMPT = """You are an expert software analyst helping a developer understand their GitHub starred repositories.
Your job is to write clear, honest, imaginative summaries that expand the developer's sense of what's possible.
Don't just describe what a repo does — help the developer see unexpected use cases, combinations with other tools, 
and creative applications they might not have considered. Be concrete, not vague."""

def summarise_repo(name, description, language, topics, readme):
    """Send repo info to Claude and get a structured summary back."""
    content = f"""Repo: {name}
Language: {language or 'Unknown'}
Topics: {', '.join(topics) if topics else 'None listed'}
GitHub description: {description or 'None provided'}

README excerpt:
{readme if readme else '[No README available]'}

Return ONLY a JSON object with exactly these fields:
{{
  "summary": "2-3 sentence plain English explanation of what this repo actually is and how it works",
  "use_cases": ["3-5 specific, concrete use cases — include at least one unexpected or creative application"],
  "category": "MUST be exactly one of these values — no variations, no hyphens added, no new values invented: CLI-Tool, Library, Framework, Boilerplate, AI-ML, Automation, Data, DevOps, Frontend, Backend, Full-Stack, Learning-Resource, API-Integration, Security, Other",
  "complexity": "one of: Beginner, Intermediate, Advanced",
  "standalone": true or false (can it be used on its own, or is it a dependency/plugin?)
}}
Return raw JSON only. No markdown, no backticks, no preamble."""

    try:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}]
        )
        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        parsed["category"] = _normalise_category(parsed.get("category", "Other"))
        return parsed
    except json.JSONDecodeError:
        return {
            "summary": "Could not parse summary.",
            "use_cases": [],
            "category": "Other",
            "complexity": "Unknown",
            "standalone": None
        }
    except Exception as e:
        print(f"    ⚠ Anthropic error: {e}")
        return None

# ── Progress / resume ──────────────────────────────────────────────────────────
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}

def save_progress(indexed):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(indexed, f)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    if not GITHUB_TOKEN or not ANTHROPIC_KEY:
        print("ERROR: Set GITHUB_TOKEN and ANTHROPIC_API_KEY environment variables.")
        return

    stars = get_all_stars()
    progress = load_progress()  # {repo_full_name: indexed_entry}
    
    total = len(stars)
    skipped = 0
    processed = 0
    failed = 0

    for i, star in enumerate(stars):
        repo = star.get("repo", star)  # handle both star+json and plain formats
        full_name = repo["full_name"]

        if full_name in progress:
            skipped += 1
            continue

        owner, repo_name = full_name.split("/", 1)
        print(f"[{i+1}/{total}] {full_name}")

        # Fetch README
        readme = get_readme(owner, repo_name)
        time.sleep(RATE_LIMIT_PAUSE)

        # AI summary
        ai = summarise_repo(
            name=full_name,
            description=repo.get("description", ""),
            language=repo.get("language", ""),
            topics=repo.get("topics", []),
            readme=readme
        )

        if ai is None:
            print(f"    ✗ Failed, will retry next run")
            failed += 1
            time.sleep(5)
            continue

        entry = {
            "name": repo_name,
            "full_name": full_name,
            "url": repo["html_url"],
            "language": repo.get("language"),
            "topics": repo.get("topics", []),
            "github_description": repo.get("description"),
            "stars": repo.get("stargazers_count", 0),
            "starred_at": star.get("starred_at", ""),
            "summary": ai["summary"],
            "use_cases": ai["use_cases"],
            "category": ai["category"],
            "complexity": ai["complexity"],
            "standalone": ai["standalone"],
            "indexed_at": datetime.now(timezone.utc).isoformat()
        }

        progress[full_name] = entry
        save_progress(progress)
        processed += 1

        print(f"    ✓ {ai['category']} | {ai['complexity']} | {ai['summary'][:80]}...")
        time.sleep(BATCH_PAUSE)

    # Write final output
    final = list(progress.values())
    with open(OUTPUT_FILE, "w") as f:
        json.dump(final, f, indent=2)

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Done.
  Total repos : {total}
  Processed   : {processed}
  Skipped     : {skipped} (already indexed)
  Failed      : {failed} (re-run to retry)
  Output      : {OUTPUT_FILE}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

if __name__ == "__main__":
    main()
