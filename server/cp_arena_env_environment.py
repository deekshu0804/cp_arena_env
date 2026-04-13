import typing
import uuid
import json
import random
import os
from collections import defaultdict
from openenv.core.env_server.interfaces import Environment  # type: ignore
from openenv.core.env_server.types import State  # type: ignore

try:
    from models import CpArenaEnvAction, CpArenaEnvObservation, CpArenaEnvState  # type: ignore
except ImportError:
    from ..models import CpArenaEnvAction, CpArenaEnvObservation, CpArenaEnvState  # type: ignore

DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "problems.json"))

def load_problems():
    with open(DATASET_PATH, "r") as f:
        return json.load(f)

PROBLEMS = load_problems()

# ─── Action constants ─────────────────────────────────────────────────────────
INFO_ACTIONS      = {0, 1, 2, 3, 4}
REASONING_ACTIONS = {5, 6, 7, 8, 9, 10}
SUBMIT_ACTIONS    = {11, 12, 13}
HINT_ACTION       = 14          # NEW: reveal_statement_hint

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

TASKS = ["algorithm_selection", "complexity_optimization", "problem_classification"]

# ─── Global leaderboard (in-memory, lives for server lifetime) ────────────────
# Structure: {task: [{"episode_id", "score", "steps", "reward", "success"}, ...]}
_LEADERBOARD: typing.Dict[str, typing.List[typing.Dict]] = defaultdict(list)
_TASK_STATS:  typing.Dict[str, typing.Dict] = {}   # aggregated stats per task


def record_episode(task: str, episode_id: str, total_reward: float,
                   steps: int, success: bool, score: float):
    """Called at episode end to update leaderboard."""
    entry = {
        "episode_id":   episode_id,
        "task":         task,
        "total_reward": round(total_reward, 3),
        "score":        round(score, 4),
        "steps":        steps,
        "success":      success,
    }
    _LEADERBOARD[task].append(entry)

    # Keep only top-50 per task by score
    _LEADERBOARD[task].sort(key=lambda x: x["score"], reverse=True)
    if len(_LEADERBOARD[task]) > 50:
        _LEADERBOARD[task] = _LEADERBOARD[task][:50]

    # Recompute aggregated stats
    all_entries = _LEADERBOARD[task]
    n = len(all_entries)
    _TASK_STATS[task] = {
        "total_episodes": n,
        "solve_rate":     round(sum(1 for e in all_entries if e["success"]) / n, 3) if n else 0.0,
        "avg_reward":     round(sum(e["total_reward"] for e in all_entries) / n, 3) if n else 0.0,
        "avg_steps":      round(sum(e["steps"] for e in all_entries) / n, 2) if n else 0.0,
        "best_score":     round(max(e["score"] for e in all_entries), 4) if n else 0.0,
        "top_10":         all_entries[:10],
    }


def get_leaderboard() -> typing.Dict:
    return {
        "leaderboard": dict(_LEADERBOARD),
        "stats":       dict(_TASK_STATS),
    }


class CpArenaEnvEnvironment(Environment):

    def __init__(self):
        super().__init__()
        self._state: typing.Any = CpArenaEnvState()       # type: ignore
        self._obs: typing.Any   = CpArenaEnvObservation() # type: ignore
        self._problem   = None
        self._total_reward      = 0.0
        self._last_tested_algo: typing.Optional[str] = None
        self._used_actions: typing.Set = set()
        self._task  = "algorithm_selection"
        self._hint_revealed     = False  # NEW: track hint usage

    # ─── Reset ────────────────────────────────────────────────────────────────

    def reset(self, seed=None, task="algorithm_selection") -> CpArenaEnvObservation:
        if seed is not None:
            random.seed(seed)

        if task not in TASKS:
            task = "algorithm_selection"

        self._task             = task
        self._problem          = random.choice(PROBLEMS)
        self._total_reward     = 0.0
        self._last_tested_algo = None
        self._used_actions     = set()
        self._hint_revealed    = False

        assert self._problem is not None
        self._state = CpArenaEnvState(  # type: ignore
            episode_id=str(uuid.uuid4()),
            step_count=0,
            problem_id=self._problem["id"],
            total_reward=0.0,
            normalized_score=0.0,
        )

        task_descriptions = {
            "algorithm_selection":   "Identify the correct algorithm for this problem.",
            "complexity_optimization": "Choose the optimal time complexity for this problem.",
            "problem_classification": "Classify this problem using minimum information reveals.",
        }

        initial_msg = (
            f"Task: {task}. {task_descriptions[task]} "
            f"Problem difficulty: {self._problem['difficulty']}. "
            f"Hint available via action 14 (reveal_statement_hint) — costs 5 pts."
        )
        self._obs = CpArenaEnvObservation(  # type: ignore
            time_remaining=300,
            attempts_left=3,
            last_verdict="none",
            last_reward=0.0,
            step_count=0,
            info_actions_taken=0,
            done=False,
        )
        return self._build_obs(0.0, False, initial_msg)

    # ─── Step ─────────────────────────────────────────────────────────────────

    def step(self, action: CpArenaEnvAction) -> CpArenaEnvObservation:
        a       = action.action_id
        reward  = 0.0
        done    = False
        message = ""
        p       = self._problem
        if p is None:
            return self._obs

        # ── NEW: Action 14 — reveal_statement_hint ───────────────────────────
        if a == HINT_ACTION:
            if self._hint_revealed:
                reward  = -3.0
                message = "Hint already revealed. Small penalty."
            else:
                reward              = -5.0
                self._hint_revealed = True
                hint = p.get("statement_hint", "")
                message = f"Problem hint: {hint}" if hint else "No hint available for this problem."
            self._obs.time_remaining -= 8
            return self._build_obs(reward, False, message)

        # ── Repeated info action penalty ─────────────────────────────────────
        if a in INFO_ACTIONS and a in self._used_actions:
            reward  = -5.0
            message = "Already revealed. Penalty applied."
            return self._build_obs(reward, done, message)

        # ── INFO actions ─────────────────────────────────────────────────────
        if a in INFO_ACTIONS:
            cost_table = [2, 4, 8, 16, 32]
            cost       = cost_table[min(self._obs.info_actions_taken, 4)]
            reward     = -cost
            self._obs.info_actions_taken += 1
            self._used_actions.add(a)

            if   a == 0:
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
                example = p.get("example_hint", "Example I/O revealed.")
                message = example

        # ── REASONING actions ────────────────────────────────────────────────
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

            if   a == 5:  self._obs.test_greedy_signal        = signal
            elif a == 6:  self._obs.test_dp_signal            = signal
            elif a == 7:  self._obs.test_graph_signal         = signal
            elif a == 8:  self._obs.test_binary_search_signal = signal
            elif a == 9:  self._obs.test_math_signal          = signal
            elif a == 10: self._obs.test_string_signal        = signal

        # ── SUBMIT actions ───────────────────────────────────────────────────
        elif a in SUBMIT_ACTIONS:
            if self._obs.attempts_left <= 0:
                reward = -50.0
                done   = True
                message = "No attempts left."
            else:
                complexity = COMPLEXITY_MAP[a]
                self._obs.attempts_left -= 1
                reward -= 10.0  # submission cost

                algo_correct = (
                    self._last_tested_algo is not None
                    and self._last_tested_algo in p["valid_algorithms"]
                )
                complexity_correct = complexity in p["valid_complexities"]

                # Consistency check
                last_algo = self._last_tested_algo
                if last_algo is not None:
                    valid_c = ALGO_COMPLEXITY_VALID.get(last_algo, [])
                    if complexity not in valid_c:
                        reward  -= 5.0
                        message += "Inconsistent complexity. "

                # ── Task 1: Algorithm Selection ──────────────────────────────
                if self._task == "algorithm_selection":
                    if algo_correct:
                        time_bonus = (self._obs.time_remaining / 300) * 30
                        reward += 100.0 + time_bonus
                        self._obs.last_verdict = "AC"
                        done    = True
                        message += f"CORRECT algorithm! Score: {self._compute_score():.3f}"
                    else:
                        reward += -30.0
                        self._obs.last_verdict = "WA"
                        message += "WRONG algorithm."

                # ── Task 2: Complexity Optimization ─────────────────────────
                elif self._task == "complexity_optimization":
                    if complexity_correct:
                        efficiency = max(0, (5 - self._obs.info_actions_taken) * 5)
                        reward += 100.0 + efficiency
                        self._obs.last_verdict = "AC"
                        done    = True
                        message += f"OPTIMAL complexity! Efficiency bonus: {efficiency}"
                    elif complexity == "nlogn" and "quadratic" in p["valid_complexities"]:
                        reward += 50.0
                        self._obs.last_verdict = "PARTIAL"
                        done    = True
                        message += "Better than needed but accepted."
                    else:
                        reward += -20.0
                        self._obs.last_verdict = "TLE"
                        message += "Wrong complexity — TLE."

                # ── Task 3: Problem Classification ───────────────────────────
                elif self._task == "problem_classification":
                    if algo_correct:
                        reveals_used   = self._obs.info_actions_taken
                        efficiency_score = max(0, 5 - reveals_used)
                        reward += 60.0 + (efficiency_score * 10)
                        self._obs.last_verdict = "AC"
                        done    = True
                        message += (
                            f"Classified correctly with {reveals_used} reveals! "
                            f"Efficiency: {efficiency_score}/5"
                        )
                    else:
                        reward += -25.0
                        self._obs.last_verdict = "WA"
                        message += "Wrong classification."

                if self._obs.attempts_left == 0 and not done:
                    reward -= 50.0
                    done    = True
                    message += " All attempts used."

        # ── Time cost ─────────────────────────────────────────────────────────
        time_costs = {
            **{i: 10 for i in INFO_ACTIONS},
            **{i: 5  for i in REASONING_ACTIONS},
            **{i: 20 for i in SUBMIT_ACTIONS},
        }
        self._obs.time_remaining -= time_costs.get(a, 5)

        if self._obs.time_remaining <= 0 and not done:
            done    = True
            reward -= 20.0
            message += " Time out!"

        self._state.step_count  += 1
        self._obs.step_count     = self._state.step_count

        if self._state.step_count >= 25 and not done:
            done    = True
            message += " Max steps reached."

        self._total_reward      += reward
        self._obs.last_reward    = reward
        self._obs.done           = done
        self._obs.message        = message

        normalized = (self._total_reward + 100) / 300
        self._state.total_reward     = self._total_reward
        self._state.normalized_score = max(0.001, min(0.999, normalized))

        # ── Record episode to leaderboard on completion ───────────────────────
        if done:
            record_episode(
                task       = self._task,
                episode_id = self._state.episode_id,
                total_reward = self._total_reward,
                steps      = self._state.step_count,
                success    = self._state.normalized_score >= 0.5,
                score      = self._state.normalized_score,
            )

        return self._build_obs(reward, done, message)

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _compute_score(self) -> float:
        return max(0.001, min(0.999, (self._total_reward + 100) / 300))

    def _build_obs(self, reward, done, message) -> CpArenaEnvObservation:
        self._obs.last_reward = reward
        self._obs.done        = done

        problem_id = self._problem["id"] if getattr(self, "_problem", None) else "N/A"
        badge = (
            f"[Episode: {problem_id} | Task: {self._task} | "
            f"Step: {self._state.step_count}/25 | "
            f"Hint: {'used' if self._hint_revealed else 'available (action 14)'}]"
        )
        self._obs.message = f"{badge}\n{message}" if message else badge
        return self._obs

    @property
    def state(self) -> typing.Any:
        return self._state
