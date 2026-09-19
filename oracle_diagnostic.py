"""Read-only environment diagnostics without printing secret values."""
from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from pathlib import Path
from oracle_costs import MODES


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def diagnostic() -> dict:
    catalogue = Path(os.getenv("HF_CATALOGUE_PATH", "hf_catalogue.json"))
    dependencies = {name: _version(name) for name in ("anthropic", "requests", "python-dotenv")}
    return {
        "status": "ready" if all(dependencies.values()) else "missing_dependencies",
        "python": sys.version.split()[0],
        "dependencies": dependencies,
        "github_token_configured": bool(os.getenv("GITHUB_TOKEN")),
        "anthropic_key_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
        "huggingface": {
            "username_configured": bool(os.getenv("HF_USERNAME")),
            "token_configured": bool(os.getenv("HF_TOKEN")),
            "catalogue_path": str(catalogue),
            "catalogue_present": catalogue.is_file(),
        },
        "source_limits": {
            "default_mode": "standard",
            "modes": MODES,
            "huggingface_results_per_type": os.getenv("ORACLE_HF_RESULTS", "2"),
        },
        "cost_control": {"default_budget_usd": 1.0, "default_model_profile": "quality",
                         "economical_quality_live_verified": False},
        "secrets_printed": False,
    }


def main() -> int:
    print(json.dumps(diagnostic(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
