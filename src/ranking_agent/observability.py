"""Logging setup for the ranking service."""

from __future__ import annotations

import logging
import sys

import structlog

from ranking_agent.models.trace import RunTrace


class _FlushStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every emit — required for live agent lines."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def configure_logging(level: str = "INFO", *, json_output: bool = False) -> None:
    """Attach a visible handler to ``ranking_agent.*`` without fighting uvicorn."""
    level_value = getattr(logging, level.upper(), logging.INFO)

    pkg = logging.getLogger("ranking_agent")
    pkg.setLevel(level_value)
    pkg.handlers.clear()
    handler = _FlushStreamHandler(sys.stderr)
    handler.setLevel(level_value)
    handler.setFormatter(logging.Formatter("%(message)s"))
    pkg.addHandler(handler)
    pkg.propagate = False  # avoid uvicorn/fastapi-cli root-logger interference

    for noisy in ("httpx", "httpcore", "httpcore.connection", "httpcore.http11", "pypdf"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level_value),
        cache_logger_on_first_use=True,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


def get_logger(name: str = "ranking_agent") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def log_trace(trace: RunTrace, *, logger_name: str = "ranking_agent.trace") -> None:
    """One-line summary of a finished candidate trace (spans already logged live)."""
    from ranking_agent.progress import say

    summary = trace.summary()
    say(
        "trace complete — spans=%s duration_ms=%s cost=$%.4f"
        % (
            summary.get("spans", len(trace.spans)),
            summary.get("duration_ms", trace.total_duration_ms),
            float(summary.get("cost_usd", 0.0) or 0.0),
        )
    )
