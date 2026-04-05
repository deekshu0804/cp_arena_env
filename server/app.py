# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Cp Arena Env Environment.

This module creates an HTTP server that exposes the CpArenaEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app  # type: ignore
except Exception:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    )

try:
    from ..models import CpArenaEnvAction, CpArenaEnvObservation  # type: ignore
    from .cp_arena_env_environment import CpArenaEnvEnvironment  # type: ignore
except ImportError:
    from models import CpArenaEnvAction, CpArenaEnvObservation  # type: ignore
    from server.cp_arena_env_environment import CpArenaEnvEnvironment  # type: ignore

import os
readme_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "README.md"))

app = create_app(
    CpArenaEnvEnvironment,
    CpArenaEnvAction,
    CpArenaEnvObservation,
    env_name="cp_arena_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

from fastapi.responses import RedirectResponse

@app.get("/")
def redirect_root():
    return RedirectResponse(url="/docs")

@app.get("/web")
def redirect_web():
    return RedirectResponse(url="/docs")


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m cp_arena_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn cp_arena_env.server.app:app --workers 4
    """
    import uvicorn  # type: ignore

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
