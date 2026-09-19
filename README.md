# Stars Indexer / Relevancy Oracle

Describe something you want to build. The Relevancy Oracle checks your saved
GitHub stars, searches for missing capabilities on GitHub and Hugging Face,
and explains which resources or smaller components could help.

## Status

Version 0.3 adds bounded source inspection and assembly advice. The Oracle reads
selected GitHub code and Hugging Face cards, configuration and source at pinned
revisions. It distinguishes whole tools, libraries, components, patterns,
models, datasets and Spaces, then proposes how eligible resources could connect.

Recommendations retain evidence, licence qualifications, uncertainty,
dependencies and tests still required. Source inspection is bounded; it is not a
complete code or security audit. Proposed assemblies are not executed or proven
ready for production. Model weights, dataset contents and Spaces are not run.

This is a source-available distribution. Operational instructions below are for
users with permission to use the software under [LICENSE](LICENSE).

## Components

| File | Role |
|---|---|
| `indexer.py` | Fetch stars and READMEs, generate summaries, resume prior work and remove entries no longer starred. |
| `health_check.py` | Add maintenance signals to the catalogue. |
| `query.py` | Legacy natural-language catalogue queries. |
| `inward_check.py` | Identify useful resources in the saved catalogue. |
| `outward_check.py` | Search GitHub gaps and screen licence and maintenance metadata. |
| `source_fetch.py` | Retrieve bounded source at pinned revisions and cache source text. |
| `source_inspection.py` | Assess reuse boundaries and validate source citations. |
| `resource_identity.py` | Keep GitHub and Hugging Face resource identities distinct. |
| `huggingface_provider.py` | Search models, datasets and Spaces; maintain optional public likes/collections; inspect evidence. |
| `assembly_advice.py` | Propose connections between eligible inspected resources, with open requirements and tests. |
| `relevancy_oracle.py` | Run the combined assessment and save its response and complete brief. |
| `render_brief.py` | Render a saved response without another AI call. |
| `oracle_diagnostic.py` | Report dependency and configuration presence without displaying secret values. |

## Setup

Use Python 3.12 or newer. Verification has covered Python 3.12.14 and 3.14.2.
Install the required libraries:

```powershell
python -m pip install anthropic requests python-dotenv
```

Create `.env` in this folder with your own credentials. Keep it private:

```dotenv
GITHUB_TOKEN=your_github_token
ANTHROPIC_API_KEY=your_anthropic_api_key
```

An optional `HF_TOKEN` enables requests using your own Hugging Face access.
Public search works without it. Pass your public Hugging Face username with
`--hf-username`; likes and collections are stored separately in
`hf_catalogue.json`. Empty public inventory is a valid result.

Run all commands from this repository's root. Leave `STARS_INDEX_PATH` unset
for the combined workflow so all scripts use the same catalogue.

## Usage

Build or refresh the GitHub catalogue, then restore its health information:

```powershell
python indexer.py
python health_check.py
```

Both steps are required after an index refresh: rebuilding the index removes
the health fields. The indexer resumes saved summaries and prunes repositories
absent from the current stars response.

Run the full Oracle, choosing `commercial` for a business or paid product:

```powershell
python relevancy_oracle.py "Your build idea" commercial --hf-username YOUR_USERNAME --refresh-hf-catalogue --output oracle-response.json --brief-output oracle-brief.txt
```

Use `personal` for personal projects. Omit the username/refresh options if you
only want public Hugging Face discovery. Add repeatable `--hf-import` options
for specific public model, dataset or Space URLs.

The saved brief includes owned resources, external candidates, source evidence,
Hugging Face resources, assembly advice, open requirements and technical handoffs.
If delivery fails, render the saved JSON without repeating paid research:

```powershell
python render_brief.py oracle-response.json
```

Refresh only public Hugging Face likes and collections, without an AI call:

```powershell
python relevancy_oracle.py --hf-username YOUR_USERNAME --refresh-hf-catalogue
```

An incomplete refresh preserves the previous complete catalogue and exits with
a failure status. Saved inventory does not replace public discovery.

Check local setup without provider calls or printing credentials:

```powershell
python relevancy_oracle.py --diagnose
```

The individual inward, outward and legacy query scripts remain available.
On Windows, use the full path to your Python executable if it is not on PATH;
`-X utf8` avoids console encoding issues. A sandboxed assistant may need its
normal command approval process to access the installed interpreter.

## Data, cost and limits

The full assessment sends catalogue-derived context and retrieved source
excerpts to Anthropic. Capability searches and public source retrieval contact
GitHub and Hugging Face. Indexing and AI assessments use your Anthropic credit;
cost depends on the selected model and the amount of material inspected.
The renderer and local diagnostic make no paid calls.

`ORACLE_MODEL` selects the Oracle's model, defaulting to `claude-opus-4-5`.
The legacy indexer and query script currently name that model directly.

- Discovery and inspection have explicit request, result and source-size bounds.
  Partial searches and uninspected candidates remain visible as limitations.
- Metadata-only, uncertain, rejected, gated or licence-incompatible resources
  cannot count as satisfied assembly requirements.
- Assembly advice remains unexecuted and can leave requirements open. It needs
  implementation and integration tests before any deployment claim.
- The final GitHub index write is not atomic. Preserve your local catalogue
  before maintenance if you need a recovery copy.
- Some scripts support `STARS_INDEX_PATH` while others use relative filenames;
  run from the root with the variable unset.
- The brief's regeneration-age note can use file modification time, which a
  health check also updates. It does not prove every AI summary is fresh.

Keep credentials, catalogues, source caches and generated reports private.
They are not included in this distribution. Assistant skills customised for a
particular user's machine are also not included.

## Licence

Source-available / view-only. See [LICENSE](LICENSE). This code may be read and
discussed but not used, copied, modified, or distributed without written
permission.
