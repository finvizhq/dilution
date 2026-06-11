import contextvars
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_ticker_var: "contextvars.ContextVar[str]" = contextvars.ContextVar(
    "log_ticker", default="-")


def set_log_ticker(ticker: str) -> None:
    """Set the ticker tag injected into subsequent log lines."""
    _ticker_var.set((ticker or "-").upper())


class _TickerFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.ticker = _ticker_var.get()
        return True

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "dilution.db"
PIPELINE_LOG_PATH = BASE_DIR / "dilution.log"
DASHBOARD_LOG_PATH = BASE_DIR / "dashboard.log"

_env_file = BASE_DIR / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

EDGAR_IDENTITY = "Peter Pagac quarkus7@gmail.com"

# ─── LLM provider switch ─────────────────────────────────────────────
# To swap models, edit LLM_PROVIDER + LLM_MODEL together. .env carries
# the matching key (XAI_API_KEY / MOONSHOT_API_KEY / GEMINI_API_KEY).
# All three providers behave identically from the extractors' POV — the
# adapter in dilution/llm_provider.py hides the differences
# (temperature/seed constraints, structured-output shape, finish_reason
# naming).
LLM_PROVIDER = "gemini"  # "xai" | "moonshot" | "gemini"
LLM_MODEL = "gemini-3.5-flash"  # production walker (promoted from gemini-3.1-flash-lite 2026-06-02 — both tiers now 3.5-flash)
# DETERMINISM NOTE: LLM_MODEL / LLM_MODEL_PERIODIC below are UNVERSIONED
# aliases. Google rotates the backing checkpoint without notice, so two
# runs days apart can hit different weights — a determinism hole that
# temperature=0 does NOT close. As of the last model-list check, Google
# published NO dated -NNN snapshot for the gemini-3.x flash models (only
# the bare alias, -preview, and -latest, all of which float). Pin both to
# a dated snapshot (e.g. "...-001") the moment one is published.
# Periodic-overhang extraction (seed + anchor reconciliation on
# 10-K/10-Q/20-F/40-F) gets a stronger model — the prompts are long,
# the schema is wide, and a missed warrant tranche poisons every
# downstream anchor diff. As of 2026-06-02 the per-filing walker
# (LLM_MODEL) was promoted from gemini-3.1-flash-lite to match this
# tier, so both now run gemini-3.5-flash and this split is currently a
# no-op — kept as a seam so the cheaper lite model can be re-introduced
# for short 8-K/6-K/424B forms if per-form cost becomes a concern (the
# eval showed lite ≈ 3.5-flash on accuracy). Only consulted when
# LLM_PROVIDER routes through make_chat()'s default model param.
LLM_MODEL_PERIODIC = "gemini-3.5-flash"
# Default xai pair: LLM_PROVIDER = "xai", LLM_MODEL = "grok-4-1-fast-non-reasoning"
# Default moonshot pair: LLM_PROVIDER = "moonshot", LLM_MODEL = "kimi-k2.6"

XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
MOONSHOT_API_KEY = os.environ.get("MOONSHOT_API_KEY", "")
MOONSHOT_BASE_URL = os.environ.get(
    "MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE_URL = os.environ.get(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/")
# Gemini service tier. "flex" is ~50% cheaper than "standard" but has
# variable latency and may return 503/429 under load (openai-python's
# default client retries those automatically). The walker is not
# latency-critical, so flex is the right default. Set to "standard"
# via env if a run is hitting too many capacity rejections.
GEMINI_SERVICE_TIER = os.environ.get("GEMINI_SERVICE_TIER", "flex")

# Gemini context window (tokens). Both active aliases — gemini-3.5-flash
# (LLM_MODEL_PERIODIC) and gemini-3.1-flash-lite (LLM_MODEL) — report the
# same limits via the v1beta models endpoint (verified 2026-06-01):
# input 1,048,576, output 65,536. Input and output are INDEPENDENT budgets
# (a full input does not eat into the output allowance). Extractors size
# their filing-text input cap (MAX_INPUT_CHARS) and the overhang output cap
# against these. The aliases float (see the DETERMINISM NOTE above) —
# re-verify if Google publishes a dated 3.x snapshot or rotates weights.
GEMINI_INPUT_TOKEN_LIMIT = 1_048_576
GEMINI_OUTPUT_TOKEN_LIMIT = 65_536

# Kimi K2.6 thinking mode. "disabled" is faster + cheaper and closer to
# the deterministic batch contract our extractors expect; "enabled"
# trades cost/latency for stronger reasoning. Ignored for xAI.
MOONSHOT_THINKING = "disabled"  # "disabled" | "enabled"

# Max concurrent in-flight LLM calls during extract/overhang stages.
# 1 = legacy serial behavior. Both providers tolerate moderate fan-out;
# SQLite writes stay serialized in the driver thread regardless.
LLM_CONCURRENCY = int(os.environ.get("LLM_CONCURRENCY", "3"))

FINVIZ_API_KEY = os.environ.get("FINVIZ_API_KEY", "")
FINVIZ_BASE_URL = os.environ.get("FINVIZ_BASE_URL", "https://elite.finviz.com")


def setup_logging(log_path: Path, level: int = logging.INFO) -> None:
    """Configure root logger: terse stderr + rotating file at log_path.

    Idempotent — re-runs replace existing handlers so the entry points
    can call this without doubling up output.
    """
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)

    ticker_filter = _TickerFilter()

    stderr = logging.StreamHandler()
    stderr.setFormatter(logging.Formatter(
        "%(asctime)s  [%(ticker)s]  %(message)s", datefmt="%H:%M:%S"))
    stderr.addFilter(ticker_filter)
    root.addHandler(stderr)

    fileh = RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    fileh.setFormatter(logging.Formatter(
        "%(asctime)s  [%(ticker)s]  %(levelname)-7s  %(name)s  %(message)s"))
    fileh.addFilter(ticker_filter)
    root.addHandler(fileh)
