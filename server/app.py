# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the CP Arena Env Environment.

Endpoints (original):
    POST /reset   — Reset the environment
    POST /step    — Execute an action
    GET  /state   — Get current episode state
    GET  /schema  — Get action/observation schemas
    WS   /ws      — WebSocket for persistent sessions

Endpoints (new):
    GET  /leaderboard        — Top runs per task, aggregated stats
    GET  /stats              — Overall solve rates + avg reward per task
    GET  /leaderboard/{task} — Leaderboard for a single task
"""

try:
    from openenv.core.env_server.http_server import create_app  # type: ignore
except Exception:
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with 'uv sync'"
    )

try:
    from ..models import CpArenaEnvAction, CpArenaEnvObservation  # type: ignore
    from .cp_arena_env_environment import CpArenaEnvEnvironment, get_leaderboard  # type: ignore
except ImportError:
    from models import CpArenaEnvAction, CpArenaEnvObservation  # type: ignore
    from server.cp_arena_env_environment import CpArenaEnvEnvironment, get_leaderboard  # type: ignore

from fastapi import HTTPException
from fastapi.responses import JSONResponse
import os

readme_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "README.md"))

# ─── Base app from OpenEnv ────────────────────────────────────────────────────
app = create_app(
    CpArenaEnvEnvironment,
    CpArenaEnvAction,
    CpArenaEnvObservation,
    env_name="cp_arena_env",
    max_concurrent_envs=1,
)

VALID_TASKS = ["algorithm_selection", "complexity_optimization", "problem_classification"]

# ─── NEW: /leaderboard ────────────────────────────────────────────────────────
@app.get("/leaderboard")
async def leaderboard():
    """
    Returns the top episodes per task ranked by normalised score,
    plus aggregated stats (solve rate, avg reward, avg steps).

    Example response:
    {
      "leaderboard": {
        "algorithm_selection": [
          {"episode_id": "...", "score": 0.82, "steps": 4, "success": true, ...},
          ...
        ]
      },
      "stats": {
        "algorithm_selection": {
          "total_episodes": 12,
          "solve_rate": 0.75,
          "avg_reward": 87.3,
          "avg_steps": 5.2,
          "best_score": 0.91
        }
      }
    }
    """
    data = get_leaderboard()
    return JSONResponse(content=data)


# ─── NEW: /leaderboard/{task} ─────────────────────────────────────────────────
@app.get("/leaderboard/{task}")
async def leaderboard_task(task: str):
    """Returns leaderboard for a single task."""
    if task not in VALID_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task}'. Valid tasks: {VALID_TASKS}",
        )
    data = get_leaderboard()
    return JSONResponse(content={
        "task":        task,
        "leaderboard": data["leaderboard"].get(task, []),
        "stats":       data["stats"].get(task, {}),
    })


# ─── NEW: /stats ──────────────────────────────────────────────────────────────
@app.get("/stats")
async def stats():
    """
    Returns a flat summary of performance across all tasks.
    Useful for the README comparison table and quick health check.
    """
    data = get_leaderboard()
    return JSONResponse(content={
        "total_tasks":   len(VALID_TASKS),
        "per_task_stats": data["stats"],
        "note": (
            "Stats accumulate across all episodes since server start. "
            "Restart the server to reset."
        ),
    })


# ─── Entry point ──────────────────────────────────────────────────────────────
def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn  # type: ignore
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
