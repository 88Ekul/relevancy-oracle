#!/usr/bin/env python3
"""
GitHub Stars Query Tool
Search your indexed repos conversationally via the terminal.
Usage: python query.py "what have I got for webhook handling"
       python query.py --list-categories
       python query.py --category AI-ML
       python query.py --complexity Beginner
"""

import os
import json
import sys
import anthropic
from dotenv import load_dotenv

load_dotenv()

INDEX_FILE = "stars_index.json"

def load_index():
    if not os.path.exists(INDEX_FILE):
        print(f"ERROR: {INDEX_FILE} not found. Run indexer.py first.")
        sys.exit(1)
    with open(INDEX_FILE) as f:
        return json.load(f)

def list_categories(index):
    from collections import Counter
    cats = Counter(r.get("category", "Other") for r in index)
    print("\nCategories in your index:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat:<20} {count} repos")

def filter_by(index, field, value):
    results = [r for r in index if r.get(field, "").lower() == value.lower()]
    print_results(results)

def print_results(results, max=20):
    if not results:
        print("No matches found.")
        return
    print(f"\n{len(results)} repo(s) found:\n")
    for r in results[:max]:
        print(f"  ★ {r['full_name']}")
        print(f"    {r['summary']}")
        print(f"    Use cases: {' | '.join(r.get('use_cases', [])[:2])}")
        print(f"    {r['url']}\n")
    if len(results) > max:
        print(f"  ... and {len(results) - max} more. Refine your query.")

def natural_query(query, index):
    """Use Claude to find relevant repos from a natural language query."""
    ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
    if not ANTHROPIC_KEY:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    # Build a compact index for the prompt (name + summary + use cases)
    compact = []
    for r in index:
        compact.append({
            "full_name": r["full_name"],
            "url": r["url"],
            "category": r.get("category"),
            "complexity": r.get("complexity"),
            "summary": r.get("summary", ""),
            "use_cases": r.get("use_cases", [])
        })

    prompt = f"""You are helping a developer search their personal GitHub stars index.

Their query: "{query}"

Here is their full index (JSON):
{json.dumps(compact, indent=1)}

Return ONLY a JSON array of the most relevant matches. Each item:
{{
  "full_name": "...",
  "url": "...",
  "relevance": "one sentence explaining exactly why this matches the query",
  "summary": "..."
}}

Return the top 5-10 most relevant. If nothing is relevant, return an empty array [].
Raw JSON only. No markdown, no backticks."""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        results = json.loads(raw.strip())
        if not results:
            print("No relevant repos found for that query.")
            return
        print(f"\n{len(results)} relevant repo(s) for: '{query}'\n")
        for r in results:
            print(f"  ★ {r['full_name']}")
            print(f"    Why: {r['relevance']}")
            print(f"    {r.get('summary', '')}")
            print(f"    {r['url']}\n")
    except json.JSONDecodeError:
        print("Could not parse response. Raw output:")
        print(raw)

def main():
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        sys.exit(0)

    index = load_index()

    if "--list-categories" in args:
        list_categories(index)
    elif "--category" in args:
        val = args[args.index("--category") + 1]
        filter_by(index, "category", val)
    elif "--complexity" in args:
        val = args[args.index("--complexity") + 1]
        filter_by(index, "complexity", val)
    elif "--list-all" in args:
        print_results(index, max=len(index))
    else:
        query = " ".join(args)
        natural_query(query, index)

if __name__ == "__main__":
    main()
