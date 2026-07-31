# Stars Indexer / Relevancy Oracle
An LLM-powered retrieval and relevance engine over your GitHub starred
repositories. It turns a large, forgotten library of stars into a queryable
answer to one question: **"I want to build X — what do I already have, and what
do I need?"**
The indexer pulls your starred repos via the GitHub API, fetches their READMEs,
and uses Claude to generate plain-English summaries and use cases, writing
everything to a local index. The oracle then reasons over that index (inward)
and searches GitHub live for genuine gaps (outward), with a human-readable brief
rendered over the combined result.
## Status
**Oracle core: complete.** Both halves join into one command and the brief
renders over the combined output.
**Honesty / verification layer: in progress.** The project's guiding principle
is that the oracle *triages* repositories on metadata — it does **not** audit
their code, and its output must never be mistaken for a safety endorsement.
Work to close that gap (direct LICENSE-file verification instead of trusting
GitHub's auto-detected metadata, explicit "not independently audited"
disclosure, and dependency-vulnerability signals) is designed but **not yet
shipped**. See Known issues below for current limitations.
## What's here
- `indexer.py` — pulls your stars, fetches READMEs, generates Claude summaries →
  `stars_index.json` (resumable via `stars_progress.json`)
- `query.py` — keyword / natural-language search over the index
- `inward_check.py` — the inward oracle: which repos you already own relate to a
  build idea
- `outward_check.py` — the outward oracle: live GitHub search for gaps, with
  licence and maintenance vetting
- `health_check.py` — enriches the index with maintenance signals
  (`pushed_at` / `archived`)
- `relevancy_oracle.py` — joins the inward and outward halves into one envelope
- `render_brief.py` — a deterministic plain-language brief over the envelope
  (no LLM call)
## Setup
Requires Python 3.12+.
    pip install anthropic requests python-dotenv
Create a `.env` file in the repo root (never commit this):
    GITHUB_TOKEN=your_github_personal_access_token
    ANTHROPIC_API_KEY=your_anthropic_api_key
    STARS_INDEX_PATH=stars_index.json
The GitHub token needs `repo` and `read:user` scopes to read your starred list.
## Usage
Run all commands from the repo root (see Known issues).
    # 1. Build the index (first run indexes everything; later runs add new stars only)
    python indexer.py
    # 2. Enrich with maintenance health (cheap, re-run any time)
    python health_check.py
    # 3. Ask the oracle: what do I have, and what do I need?
    python relevancy_oracle.py "I want to build X" personal
    python relevancy_oracle.py "I want to build X" commercial
    # Render the human-readable brief over the result
    python relevancy_oracle.py "I want to build X" personal | python render_brief.py
    # Or run the halves individually
    python inward_check.py "I want to build X"
    python outward_check.py "I want to build X" personal
    # Original keyword query (still works)
    python query.py "what have I got for webhook handling"
## How it works
The inward half sends a compacted view of your index to Claude and asks which
owned repos relate to the build idea — including non-obvious connections, not
just keyword matches. The outward half identifies capability gaps, searches
GitHub live for each, and vets candidates on licence and maintenance before
recommending them. Repos whose licence can't be verified are flagged rather than
silently dropped. The brief renderer is deterministic — every judgement it shows
is already in the data.
## Cost
Roughly $1–3 to index 200–300 repos, depending on README length. Subsequent runs
(new stars only) cost pennies.
## Known issues
This project is under active development. Current known issues, tracked privately:
- The final index write in `indexer.py` is not atomic — an interruption during
  the write can corrupt `stars_index.json` (fix scheduled).
- `indexer.py` and `query.py` use a hardcoded relative index path while the
  oracle modules read `STARS_INDEX_PATH` from the environment — run both from
  the repo root until fixed.
- The brief's staleness warning keys off file mtime, which `health_check.py`
  refreshes on every run — treat the "index last regenerated" note as unreliable
  for now.
- Repo-count discrepancy between the indexer and health-check counts, under
  diagnosis.
## Licence
Source-available / view-only. See [LICENSE](LICENSE). This code may be read and
discussed but not used, copied, modified, or distributed without written
permission.
