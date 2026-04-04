from typing import Optional
from openenv.core.env_server import Action, Observation, State  # type: ignore


# ─────────────────────────────────────────
# ACTION
# ─────────────────────────────────────────
class CpArenaEnvAction(Action):
    """
    Single integer action (0-13).

    INFO GATHERING (costs time):
        0  reveal_N
        1  reveal_time_limit
        2  reveal_memory
        3  reveal_tags
        4  reveal_example

    REASONING (returns signal, costs -1):
        5  test_greedy
        6  test_dp
        7  test_graph
        8  test_binary_search
        9  test_math
        10 test_string

    SUBMISSION (costs time + attempt):
        11 submit_linear      O(N)
        12 submit_nlogn       O(N log N)
        13 submit_quadratic   O(N²)
    """
    action_id: int = 0


# ─────────────────────────────────────────
# OBSERVATION
# ─────────────────────────────────────────
class CpArenaEnvObservation(Observation):
    # revealed info (None = not yet revealed)
    revealed_n: Optional[str] = None
    revealed_time_limit: Optional[str] = None
    revealed_memory: Optional[str] = None
    revealed_tags: Optional[str] = None
    revealed_example: bool = False

    # reasoning signals (None = not tested yet)
    signal_greedy: Optional[bool] = None
    signal_dp: Optional[bool] = None
    signal_graph: Optional[bool] = None
    signal_binary_search: Optional[bool] = None
    signal_math: Optional[bool] = None
    signal_string: Optional[bool] = None

    # episode state
    time_remaining: int = 300
    attempts_left: int = 3
    last_verdict: Optional[str] = None
    last_reward: float = 0.0
    done: bool = False
    step_count: int = 0
    message: str = ""  # human-readable feedback
    task: str = "algorithm_selection" 
    episode_status: str = "Running"


# ─────────────────────────────────────────
# STATE
# ─────────────────────────────────────────
class CpArenaEnvState(State):
    episode_id: str = ""
    step_count: int = 0
    problem_id: str = ""
    total_reward: float = 0.0
    normalized_score: float = 0.0