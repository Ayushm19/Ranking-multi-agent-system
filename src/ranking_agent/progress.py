"""Progress lines for agent runs (stderr + ``logs/agent.log``)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "agent.log"


def say(message: str) -> None:
    """Emit one progress line to the terminal and ``logs/agent.log``."""
    text = message.rstrip()
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{stamp}] {text}"

    try:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    except Exception:  # noqa: BLE001 — never break a rank over logging
        pass

    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
    except Exception:  # noqa: BLE001
        pass
