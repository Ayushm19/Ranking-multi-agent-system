"""Dev entrypoint for `uv run fastapi dev`.

Re-exports the FastAPI app so the CLI can find it without a module path.
"""

from ranking_agent.api.app import app

__all__ = ["app"]
