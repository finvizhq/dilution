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
# the matching key (XAI_API_KEY or MOONSHOT_API_KEY). The two providers
# behave identically from the extractors' POV — the adapter in
# dilution/llm_provider.py hides the differences (temperature/seed
# constraints, structured-output shape, finish_reason naming).
LLM_PROVIDER = "xai"  # "xai" | "moonshot"
LLM_MODEL = "grok-4-1-fast-non-reasoning"
# Suggested when LLM_PROVIDER="moonshot": LLM_MODEL = "kimi-k2.6"

XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
MOONSHOT_API_KEY = os.environ.get("MOONSHOT_API_KEY", "")
MOONSHOT_BASE_URL = os.environ.get(
    "MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1")

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
