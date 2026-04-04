import typing
import uuid
import json
import random
import os
from openenv.core.env_server.interfaces import Environment  # type: ignore
from openenv.core.env_server.types import State  # type: ignore

try:
    from models import CpArenaEnvAction, CpArenaEnvObservation, CpArenaEnvState  # type: ignore
except ImportError:
    from ..models import CpArenaEnvAction, CpArenaEnvObservation, CpArenaEnvState  # type: ignore

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset", "problems.json")

def load_problems():
    with open(DATASET_PATH, "r") as f:
        return json.load(f)

PROBLEMS = load_problems()

# Action constants
INFO_ACTIONS = {0, 1, 2, 3, 4}
REASONING_ACTIONS = {5, 6, 7, 8, 9, 10}
SUBMIT_ACTIONS = {11, 12, 13}

ALGO_MAP = {
    5: "greedy", 6: "dp", 7: "graph",
    8: "binary_search", 9: "math", 10: "string",
}

COMPLEXITY_MAP = {11: "linear", 12: "nlogn", 13: "quadratic"}

ALGO_COMPLEXITY_VALID = {
    "greedy":        ["linear", "nlogn"],
    "dp":            ["linear", "nlogn", "quadratic"],
    "graph":         ["linear", "nlogn"],
    "binary_search": ["nlogn"],
    "math":          ["linear"],
    "string":        ["linear", "nlogn"],
    "brute_force":   ["quadratic"],
}

# Valid tasks
TASKS = ["algorithm_selection", "complexity_optimization", "problem_classification"]


class CpArenaEnvEnvironment(Environment):

    def __init__(self):
        super().__init__()
        self._state: typing.Any = CpArenaEnvState()  # type: ignore
        self._obs: typing.Any = CpArenaEnvObservation()  # type: ignore
        self._problem = None
        self._total_reward = 0.0
        self._last_tested_algo: str | None = None
        self._used_actions = set()
        self._task = "algorithm_selection"

    def reset(self, seed=None, task="algorithm_selection") -> CpArenaEnvObservation:
        if seed is not None:
            random.seed(seed)

        # Validate task
        if task not in TASKS:
            task = "algorithm_selection"

        self._task = task
        self._problem = random.choice(PROBLEMS)
        self._total_reward = 0.0
        self._last_tested_algo = None
        self._used_actions = set()

        assert self._problem is not None
        self._state = CpArenaEnvState(episode_id=str(uuid.uuid4()), step_count=0, problem_id=self._problem["id"], total_reward=0.0, normalized_score=0.0)  # type: ignore

        task_descriptions = {
            "algorithm_selection": "Identify the correct algorithm for this problem.",
            "complexity_optimization": "Choose the optimal time complexity for this problem.",
            "problem_classification": "Classify this problem using minimum information reveals.",
        }

        self._obs = CpArenaEnvObservation(time_remaining=300, attempts_left=3, last_verdict="none", last_reward=0.0, step_count=0, info_actions_taken=0, done=False, message=f"Task: {task}. {task_descriptions[task]} Problem difficulty: {self._problem['difficulty']}.")  # type: ignore
        return self._obs

    def step(self, action: CpArenaEnvAction) -> CpArenaEnvObservation:
        a = action.action_id
        reward = 0.0
        done = False
        message = ""
        p = self._problem
        if p is None:
            return self._obs

        # Repeated info action penalty
        if a in INFO_ACTIONS and a in self._used_actions:
            reward = -5.0
            message = "Already revealed. Penalty applied."
            return self._build_obs(reward, done, message)

        # --- INFO ACTIONS ---
        if a in INFO_ACTIONS:
            cost_table = [2, 4, 8, 16, 32]
            cost = cost_table[min(self._obs.info_actions_taken, 4)]
            reward = -cost
            self._obs.info_actions_taken += 1
            self._used_actions.add(a)

            if a == 0:
                self._obs.revealed_n = p["N_scale"]
                message = f"N scale: {p['N_scale']}"
            elif a == 1:
                self._obs.revealed_time_limit = p["time_pressure"]
                message = f"Time pressure: {p['time_pressure']}"
            elif a == 2:
                self._obs.revealed_memory = p["memory_pressure"]
                message = f"Memory pressure: {p['memory_pressure']}"
            elif a == 3:
                self._obs.revealed_tags = str(p["tags"])
                message = f"Tags: {p['tags']}"
            elif a == 4:
                self._obs.revealed_example = True
                message = "Example I/O revealed."

        # --- REASONING ACTIONS ---
        elif a in REASONING_ACTIONS:
            algo = ALGO_MAP[a]
            if algo in p["valid_algorithms"]:
                reward = 3.0
                signal = True
                message = f"test_{algo}: PLAUSIBLE (+3)"
            else:
                reward = -3.0
                signal = False
                message = f"test_{algo}: UNLIKELY (-3)"

            self._last_tested_algo = algo
            self._used_actions.add(a)

            if a == 5:  self._obs.test_greedy_signal = signal
            elif a == 6: self._obs.test_dp_signal = signal
            elif a == 7: self._obs.test_graph_signal = signal
            elif a == 8: self._obs.test_binary_search_signal = signal
            elif a == 9: self._obs.test_math_signal = signal
            elif a == 10: self._obs.test_string_signal = signal

        # --- SUBMIT ACTIONS ---
        elif a in SUBMIT_ACTIONS:
            if self._obs.attempts_left <= 0:
                reward = -50.0
                done = True
                message = "No attempts left."
            else:
                complexity = COMPLEXITY_MAP[a]
                self._obs.attempts_left -= 1
                reward -= 10.0  # submission cost

                algo_correct = (
                    self._last_tested_algo is not None and
                    self._last_tested_algo in p["valid_algorithms"]
                )
                complexity_correct = complexity in p["valid_complexities"]

                # Consistency check
                last_algo = self._last_tested_algo
                if last_algo is not None:
                    valid_c = ALGO_COMPLEXITY_VALID.get(last_algo, [])
                    if complexity not in valid_c:
                        reward -= 5.0
                        message += "Inconsistent complexity. "

                # ─── TASK 1: Algorithm Selection ───
                if self._task == "algorithm_selection":
                    if algo_correct:
                        time_bonus = (self._obs.time_remaining / 300) * 30
                        reward += 100.0 + time_bonus
                        self._obs.last_verdict = "AC"
                        done = True
                        message += f"CORRECT algorithm! Score: {self._compute_score():.2f}"
                    else:
                        reward += -30.0
                        self._obs.last_verdict = "WA"
                        message += "WRONG algorithm."

                # ─── TASK 2: Complexity Optimization ───
                elif self._task == "complexity_optimization":
                    if complexity_correct:
                        efficiency = max(0, (5 - self._obs.info_actions_taken) * 5)
                        reward += 100.0 + efficiency
                        self._obs.last_verdict = "AC"
                        done = True
                        message += f"OPTIMAL complexity! Efficiency bonus: {efficiency}"
                    elif complexity == "nlogn" and "quadratic" in p["valid_complexities"]:
                        # Better than needed — partial credit
                        reward += 50.0
                        self._obs.last_verdict = "PARTIAL"
                        done = True
                        message += "Better than needed but accepted."
                    else:
                        reward += -20.0
                        self._obs.last_verdict = "TLE"
                        message += "Wrong complexity — TLE."

                # ─── TASK 3: Problem Classification ───
                elif self._task == "problem_classification":
                    if algo_correct:
                        # Reward based on how few reveals were needed
                        reveals_used = self._obs.info_actions_taken
                        efficiency_score = max(0, 5 - reveals_used)
                        reward += 60.0 + (efficiency_score * 10)
                        self._obs.last_verdict = "AC"
                        done = True
                        message += f"Classified correctly with {reveals_used} reveals! Efficiency: {efficiency_score}/5"
                    else:
                        reward += -25.0
                        self._obs.last_verdict = "WA"
                        message += "Wrong classification."

                if self._obs.attempts_left == 0 and not done:
                    reward -= 50.0
                    done = True
                    message += " All attempts used."

        # Time cost
        time_costs = {
            **{i: 10 for i in INFO_ACTIONS},
            **{i: 5 for i in REASONING_ACTIONS},
            **{i: 20 for i in SUBMIT_ACTIONS}
        }
        self._obs.time_remaining -= time_costs.get(a, 5)

        if self._obs.time_remaining <= 0 and not done:
            done = True
            reward -= 20.0
            message += " Time out!"

        self._state.step_count += 1
        self._obs.step_count = self._state.step_count

        if self._state.step_count >= 25 and not done:
            done = True
            message += " Max steps reached."

        self._total_reward += reward
        self._obs.last_reward = reward
        self._obs.done = done
        self._obs.message = message

        normalized = (self._total_reward + 100) / 300
        self._state.total_reward = self._total_reward
        self._state.normalized_score = max(0.0, min(1.0, normalized))

        return self._build_obs(reward, done, message)

    def _compute_score(self):
        return max(0.0, min(1.0, (self._total_reward + 100) / 300))

    def _build_obs(self, reward, done, message) -> CpArenaEnvObservation:
        self._obs.last_reward = reward
        self._obs.done = done
        self._obs.message = message
        return self._obs

    @property
    def state(self) -> typing.Any:
        return self._state