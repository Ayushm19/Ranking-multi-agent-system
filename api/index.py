"""Vercel Python Serverless Function entrypoint — re-exports the FastAPI app."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ranking_agent.api.app import app  # noqa: E402

__all__ = ["app"]
