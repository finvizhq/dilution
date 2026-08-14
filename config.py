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
# run_inspect.py (the walker debug view) logs here, keeping its request
# noise out of the pipeline log.
DASHBOARD_LOG_PATH = BASE_DIR / "dashboard.log"

_env_file = BASE_DIR / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

EDGAR_IDENTITY = "Peter Pagac quarkus7@gmail.com"


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")  # "" = SDK default
OPENAI_SERVICE_TIER = os.environ.get("OPENAI_SERVICE_TIER", "flex")
OPENAI_REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "low")

LLM_MODEL = "gpt-5.6-luna"  # per-filing walker (8-K/6-K/424B/S-1)
LLM_MODEL_PERIODIC = "gpt-5.6-terra"

OPENAI_MAX_INPUT_TOKENS = 922_000
OPENAI_MAX_OUTPUT_TOKENS = 128_000
# Densest tokenization observed on SEC markdown. Deliberately a floor
# (real text runs 3.5-4 chars/token) so the char cap derived from it
# cannot overshoot the token ceiling.
CHARS_PER_TOKEN_FLOOR = 3

# Max concurrent in-flight LLM calls during extract/overhang stages.
# 1 = legacy serial behavior. OpenAI tolerates moderate fan-out;
# SQLite writes stay serialized in the driver thread regardless.
LLM_CONCURRENCY = int(os.environ.get("LLM_CONCURRENCY", "3"))

FINVIZ_API_KEY = os.environ.get("FINVIZ_API_KEY", "")
FINVIZ_BASE_URL = os.environ.get("FINVIZ_BASE_URL", "https://elite.finviz.com")
FINVIZ_INGEST_TOKEN = os.environ.get("FINVIZ_INGEST_TOKEN", "")


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
