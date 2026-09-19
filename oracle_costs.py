"""Oracle-only spending guard, usage journal and revision-aware response reuse.

Prices are published USD list prices, not an invoice. Token-count estimates
can differ from billed usage; the reservation includes a margin and maximum
output. Provider account limits remain the final billing backstop.
"""
from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import uuid

PRICING_DATE = "2026-09-19"
PRICING_URL = "https://platform.claude.com/docs/en/about-claude/pricing"
# Base input, output, 5-minute cache creation, 1-hour creation, cache read / million.
PRICES = {
    "claude-opus-4-5": (5, 25, 6.25, 10, .5),
    "claude-opus-4-5-20251101": (5, 25, 6.25, 10, .5),
    "claude-sonnet-4-6": (3, 15, 3.75, 6, .3),
    "claude-sonnet-4-5-20250929": (3, 15, 3.75, 6, .3),
    "claude-haiku-4-5-20251001": (1, 5, 1.25, 2, .1),
}
MODES = {
    "quick": {"github_repositories": 0, "inward_repositories": 0, "hf_assessments": 0, "gaps": 0},
    "standard": {"github_repositories": 4, "inward_repositories": 2, "hf_assessments": 3, "gaps": 3},
    "deep": {"github_repositories": 6, "inward_repositories": 2, "hf_assessments": 6, "gaps": 5},
}
_ACTIVE = ContextVar("oracle_cost_session", default=None)


class CostControlError(RuntimeError):
    """A safe stop which must not be swallowed by provider fallbacks."""


class BudgetExceeded(CostControlError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def _plain(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, SimpleNamespace):
        return _plain(vars(value))
    return value


def _response(value, replay=False):
    # Keep tool input dictionaries intact: only SDK envelope/block attributes
    # are consumed by the existing parsers and evidence validators.
    result = SimpleNamespace(**value)
    result.content = [SimpleNamespace(**block) for block in value.get("content", [])]
    usage = dict(value.get("usage") or {})
    if replay:
        usage = {"input_tokens": 0, "output_tokens": 0,
                 "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    result.usage = SimpleNamespace(**usage)
    result._oracle_replayed = replay
    return result


def usage_cost(model, usage):
    if model not in PRICES:
        raise CostControlError("Model has no verified price; add a reviewed price before use.")
    rates = PRICES[model]
    def count(name, source=usage):
        value = source.get(name, 0)
        if value is None and name not in ('input_tokens', 'output_tokens'):
            value = 0  # The SDK serialises absent optional cache counters as null.
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CostControlError("Provider returned invalid token usage.")
        return value
    if "input_tokens" not in usage or "output_tokens" not in usage:
        raise CostControlError("Provider returned no complete usage record.")
    creation = count("cache_creation_input_tokens")
    split = usage.get("cache_creation") or {}
    hour = count("ephemeral_1h_input_tokens", split)
    five = count("ephemeral_5m_input_tokens", split)
    if hour + five > creation:
        raise CostControlError("Inconsistent cache usage record.")
    # Unknown cache lifetime gets the higher price, never understated.
    return (count("input_tokens") * rates[0] + count("output_tokens") * rates[1]
            + five * rates[2] + (creation - five) * rates[3]
            + count("cache_read_input_tokens") * rates[4]) / 1_000_000


def current_session():
    return _ACTIVE.get()


def mode_limit(name, fallback):
    session = current_session()
    return MODES[session.mode][name] if session else fallback


def message_call(client, stage, *, reusable=False, **kwargs):
    session = current_session()
    if session is None:
        raise CostControlError("Paid Oracle calls require a RunSession; use relevancy_oracle.py.")
    return session.call(client, stage, reusable=reusable, **kwargs)


class RunSession:
    """One durable total budget, including earlier attempts when resuming."""
    def __init__(self, request, *, mode="standard", max_cost_usd=1.0,
                 model_profile="quality", run_dir=None, resume=False,
                 cache_dir=None, catalogue_paths=()):
        if mode not in MODES or model_profile not in ("quality", "economical"):
            raise ValueError("Unknown Oracle mode or model profile.")
        if not isinstance(max_cost_usd, (int, float)) or isinstance(max_cost_usd, bool) or not math.isfinite(max_cost_usd) or max_cost_usd < 0:
            raise ValueError("Budget must be a finite non-negative USD amount.")
        self.mode, self.profile = mode, model_profile
        self.cap = float(max_cost_usd)
        self.path = Path(run_dir) if run_dir else Path("oracle-runs") / ("run-" + uuid.uuid4().hex)
        self.cache = Path(cache_dir) if cache_dir else Path(".oracle-assessment-cache")
        self.resume = resume
        self.base_model = os.getenv("ORACLE_MODEL", "claude-opus-4-5")
        root = Path(__file__).resolve().parent
        self.code_hash = digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(root.glob("*.py"))})
        files = {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                 if Path(p).is_file() else None for p in catalogue_paths}
        self.identity = digest({"request": request, "mode": mode, "profile": model_profile,
                                "model": self.base_model, "files": files, "code": self.code_hash,
                                "settings": {k: os.getenv(k) for k in ('HF_USERNAME', 'ORACLE_HF_RESULTS')},
                                "prices": PRICES})
        self.data = None
        self._token = None

    def __enter__(self):
        self.path.mkdir(parents=True, exist_ok=True)
        try:
            self.lock_fd = os.open(self.path / "run.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise CostControlError("Run is locked. Check that no process is running before removing a stale run.lock.") from None
        try:
            os.write(self.lock_fd, str(os.getpid()).encode())
            ledger = self.path / "usage.json"
            if self.resume:
                self.data = json.loads(ledger.read_text(encoding="utf-8"))
                if self.data["identity"] != self.identity:
                    raise CostControlError("Request, catalogues, code or settings changed; start a new run.")
                if self.cap < self.committed:
                    raise CostControlError("Resume budget is below money already recorded/reserved.")
            else:
                if ledger.exists():
                    raise CostControlError("Run already exists; use --resume to retain its budget and checkpoints.")
                self.data = {"schema": 1, "identity": self.identity, "started_at": self.now(),
                             "mode": self.mode, "model_profile": self.profile, "calls": [],
                             "replays": [], "stages": {}, "status": "running"}
            self.data.update(budget_usd=self.cap, status="running")
            self.save()
            self._token = _ACTIVE.set(self)
            return self
        except BaseException:
            os.close(self.lock_fd)
            (self.path / "run.lock").unlink(missing_ok=True)
            raise

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type:
                self.data["status"] = "interrupted"
                self.save()
        finally:
            _ACTIVE.reset(self._token)
            os.close(self.lock_fd)
            (self.path / "run.lock").unlink(missing_ok=True)

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    def save(self):
        atomic_json(self.path / "usage.json", self.data)

    @property
    def committed(self):
        return sum(c.get("cost_usd", c["reserved_usd"]) for c in self.data["calls"])

    def model(self, stage):
        if self.profile == "economical":
            if stage in ("gaps", "vetting", "hf_plan", "source_select"):
                return "claude-haiku-4-5-20251001"
            if stage == "inward":
                return "claude-sonnet-4-6"
        return self.base_model

    def stage(self, name, action):
        record = self.data["stages"].get(name)
        if record:
            return json.loads(json.dumps(record["result"]))
        result = action()
        self.data["stages"][name] = {"completed_at": self.now(), "result": _plain(result)}
        self.save()
        return json.loads(json.dumps(result))

    def call(self, client, stage, *, reusable=False, **kwargs):
        kwargs["model"] = self.model(stage)
        model = kwargs["model"]
        if model not in PRICES:
            raise CostControlError("Model has no reviewed price; no paid call was made.")
        key = digest({"schema": 1, "code": self.code_hash, "stage": stage, "request": kwargs})
        for record in reversed(self.data["calls"]):
            if record["key"] == key:
                if record["status"] == "complete":
                    return self._replay(stage, key, record["response"], "checkpoint", record.get('completed_at'))
                raise CostControlError("This call has uncertain billing; its reservation is retained. It will not be retried automatically.")
        cache_path = self.cache / (key + ".json")
        if reusable and cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached["key"] == key and cached["hash"] == digest(cached["response"]):
                    return self._replay(stage, key, cached["response"], "assessment_cache", cached.get('saved_at'))
            except (OSError, ValueError, KeyError, TypeError):
                pass  # A damaged cache is a miss; the budget still governs new work.
        if self.committed >= self.cap:
            raise BudgetExceeded("Run budget exhausted; completed work is saved.")
        # No silent SDK retries, including the free count endpoint.
        client = client.with_options(max_retries=0)
        count_args = {k: v for k, v in kwargs.items() if k != "max_tokens"}
        try:
            tokens = client.messages.count_tokens(**count_args).input_tokens
            if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
                raise ValueError("Invalid count")
        except Exception:
            raise CostControlError("Could not estimate input tokens; no paid call was made.") from None
        # 10% plus 1,024 input tokens, with all allowed output reserved.
        reserve = ((math.ceil(tokens * 1.10) + 1024) * PRICES[model][0]
                   + kwargs["max_tokens"] * PRICES[model][1]) / 1_000_000
        if self.committed + reserve > self.cap + 1e-12:
            raise BudgetExceeded(f"Next {stage} call needs a ${reserve:.4f} reservation; run budget would be exceeded.")
        record = {"key": key, "stage": stage, "model": model, "started_at": self.now(),
                  "input_token_estimate": tokens, "reserved_usd": reserve, "status": "pending"}
        self.data["calls"].append(record)
        self.save()  # Write before sending: a crash never resets the budget.
        try:
            response = client.messages.create(**kwargs)
            raw = _plain(response)
            usage = raw.get("usage") or {}
            cost = usage_cost(model, usage)
            record.update(status="complete", usage=usage, cost_usd=cost,
                          response=raw, completed_at=self.now())
            self.save()  # A completed response survives later parsing/render failures.
        except Exception as exc:
            if record["status"] != "complete":
                record.update(status="uncertain", error=type(exc).__name__)
                self.save()
            raise CostControlError("Model call failed or usage was unavailable; reserved cost retained, no automatic retry.") from None
        if reusable and raw.get("stop_reason") in ("end_turn", "tool_use"):
            atomic_json(cache_path, {"key": key, "response": raw, "hash": digest(raw),
                                     "saved_at": self.now()})
        return response

    def _replay(self, stage, key, response, origin, original_at=None):
        self.data["replays"].append({"stage": stage, "key": key, "origin": origin,
                                     "original_response_at": original_at, "at": self.now()})
        self.save()
        return _response(response, replay=True)

    def report(self):
        calls = self.data["calls"]
        stages = {}
        for call in calls:
            entry = stages.setdefault(call["stage"], {"calls": 0, "recorded_usd": 0,
                                                       "uncertain_reserved_usd": 0, "input_tokens": 0,
                                                       "output_tokens": 0, "cache_input_tokens": 0})
            entry["calls"] += 1
            entry["recorded_usd"] += call.get("cost_usd", 0)
            if call["status"] != "complete":
                entry["uncertain_reserved_usd"] += call["reserved_usd"]
            usage = call.get("usage", {})
            for field in ("input_tokens", "output_tokens"):
                entry[field] += usage.get(field, 0)
            entry["cache_input_tokens"] += (usage.get("cache_read_input_tokens") or 0) + (usage.get("cache_creation_input_tokens") or 0)
        return {"mode": self.mode, "model_profile": self.profile, "budget_usd": self.cap,
                "recorded_usd": sum(c.get("cost_usd", 0) for c in calls),
                "committed_usd": self.committed, "remaining_usd": max(0, self.cap - self.committed),
                "usage_complete": all(c["status"] == "complete" for c in calls),
                "api_calls": len(calls), "reused_calls": len(self.data["replays"]), "stages": stages,
                "run_directory": str(self.path.resolve()), "started_at": self.data["started_at"],
                "resumed": self.resume, "pricing_date": PRICING_DATE, "pricing_url": PRICING_URL,
                "note": "USD list-price calculation, not an invoice. Budget reservations use estimated input plus a margin and maximum output. Resumed stages retain their original evidence dates. Economical model quality has not been live-compared."}
