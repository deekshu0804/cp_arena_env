# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cp Arena Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient  # type: ignore
from openenv.core.client_types import StepResult  # type: ignore
from openenv.core.env_server.types import State  # type: ignore

from .models import CpArenaEnvAction, CpArenaEnvObservation  # type: ignore


class CpArenaEnv(
    EnvClient[CpArenaEnvAction, CpArenaEnvObservation, State]
):
    """
    Client for the Cp Arena Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with CpArenaEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.message)
        ...
        ...     result = client.step(CpArenaEnvAction(action_id=0))
        ...     print(result.observation.message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = CpArenaEnv.from_docker_image("cp_arena_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(CpArenaEnvAction(action_id=1))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: CpArenaEnvAction) -> Dict:
        """
        Convert CpArenaEnvAction to JSON payload for step message.

        Args:
            action: CpArenaEnvAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_id": action.action_id,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CpArenaEnvObservation]:
        """
        Parse server response into StepResult[CpArenaEnvObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with CpArenaEnvObservation
        """
        obs_data = payload.get("observation", {})
        observation = CpArenaEnvObservation(
            revealed_n=obs_data.get("revealed_n"),
            revealed_time_limit=obs_data.get("revealed_time_limit"),
            revealed_memory=obs_data.get("revealed_memory"),
            revealed_tags=obs_data.get("revealed_tags"),
            revealed_example=obs_data.get("revealed_example", False),
            signal_greedy=obs_data.get("signal_greedy"),
            signal_dp=obs_data.get("signal_dp"),
            signal_graph=obs_data.get("signal_graph"),
            signal_binary_search=obs_data.get("signal_binary_search"),
            signal_math=obs_data.get("signal_math"),
            signal_string=obs_data.get("signal_string"),
            time_remaining=obs_data.get("time_remaining", 300),
            attempts_left=obs_data.get("attempts_left", 3),
            last_verdict=obs_data.get("last_verdict"),
            last_reward=obs_data.get("last_reward", 0.0),
            done=obs_data.get("done", False),
            step_count=obs_data.get("step_count", 0),
            message=obs_data.get("message", ""),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
