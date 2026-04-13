"""
inference.py — CP-Arena Demo & Training Agent

Two modes:
  1. Q-learning (default, fully offline — no API key needed)
  2. LLM agent (optional, set USE_LLM=1 + HF_TOKEN)

Usage:
  # Offline Q-learning (runs anywhere, no token):
  python inference.py

  # LLM mode:
  USE_LLM=1 HF_TOKEN=hf_xxx python inference.py

  # Train Q-learning for N episodes then save results:
  EPISODES=200 python inference.py
"""

import asyncio
import os
import json
import random
import textwrap
import traceback
import math
from collections import defaultdict
from typing import List, Optional, Dict, Any

# ─── Config ───────────────────────────────────────────────────────────────────
USE_LLM      = os.getenv("USE_LLM", "0") == "1"
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
IMAGE_NAME   = os.getenv("IMAGE_NAME",   "cp-arena-env:latest")
BENCHMARK    = "cp_arena_env"
MAX_STEPS    = 15
EPISODES     = int(os.getenv("EPISODES", "100"))

TASKS = [
    "algorithm_selection",
    "complexity_optimization",
    "problem_classification",
]

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Logging ──────────────────────────────────────────────────────────────────

def log_start(task, env=None, model=None):
    # Exact format from reference inference.py
    print(f"[START] task={task} env={env or BENCHMARK} model={model or MODEL_NAME}", flush=True)

def log_step(step, action=None, reward=0.0, done=False, error=None):
    # Exact format from reference inference.py
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(task, success, steps, score, rewards=None):
    # score must be strictly between 0 and 1 — clamp to (0.001, 0.999)
    safe_score = max(0.001, min(0.999, float(score) if score else 0.001))
    rewards_str = ",".join(f"{r:.2f}" for r in (rewards or []))
    print(f"[END] success={str(success).lower()} steps={steps} score={safe_score:.2f} rewards={rewards_str}", flush=True)

# ─── Q-Learning Agent (fully offline) ─────────────────────────────────────────

class QLearningAgent:
    """
    Tabular Q-learning agent for CP-Arena.

    State space: (revealed_n, revealed_time, info_taken_bucket, last_verdict, attempts_left)
    Action space: 0-13 (same as environment)
    """

    # Actions grouped by type for smarter exploration
    INFO_ACTIONS      = [0, 1, 2, 3, 4]
    REASONING_ACTIONS = [5, 6, 7, 8, 9, 10]
    SUBMIT_ACTIONS    = [11, 12, 13]
    ALL_ACTIONS       = list(range(14))

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.97,
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.episode_count = 0

    # ── State encoding ────────────────────────────────────────────────────────

    def _encode_state(self, obs_dict: Dict[str, Any]) -> str:
        """Convert observation dict into a hashable state string."""
        n       = str(obs_dict.get("revealed_n") or "?")
        tl      = str(obs_dict.get("revealed_time_limit") or "?")
        mem     = str(obs_dict.get("revealed_memory") or "?")
        verdict = str(obs_dict.get("last_verdict") or "none")
        attempts = int(obs_dict.get("attempts_left") or 3)
        # Bucket info_actions_taken: 0, 1, 2, 3+
        info_bucket = min(int(obs_dict.get("info_actions_taken") or 0), 3)
        # Which reasoning signals are positive
        signals = "".join([
            "G" if obs_dict.get("test_greedy_signal") else "_",
            "D" if obs_dict.get("test_dp_signal") else "_",
            "R" if obs_dict.get("test_graph_signal") else "_",
            "B" if obs_dict.get("test_binary_search_signal") else "_",
            "M" if obs_dict.get("test_math_signal") else "_",
            "S" if obs_dict.get("test_string_signal") else "_",
        ])
        return f"{n}|{tl}|{mem}|{info_bucket}|{verdict}|{attempts}|{signals}"

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, obs_dict: Dict[str, Any], task: str) -> int:
        """ε-greedy with task-aware fallback heuristic."""
        if random.random() < self.epsilon:
            return self._heuristic_action(obs_dict, task)
        state = self._encode_state(obs_dict)
        q_vals = self.q_table[state]
        if not q_vals:
            return self._heuristic_action(obs_dict, task)
        return max(self.ALL_ACTIONS, key=lambda a: q_vals[a])

    def _heuristic_action(self, obs_dict: Dict[str, Any], task: str) -> int:
        """
        Structured exploration: mimic human strategy before random.
        Phase 1 (no N revealed): reveal N
        Phase 2 (no TL revealed): reveal time_limit
        Phase 3 (few tests done): run a reasoning test
        Phase 4: submit
        """
        n  = obs_dict.get("revealed_n")
        tl = obs_dict.get("revealed_time_limit")
        info_taken = int(obs_dict.get("info_actions_taken") or 0)

        # Always reveal N first if missing
        if not n:
            return 0
        # Then time limit
        if not tl:
            return 1

        # Check if any positive signal exists
        signals = {
            5: obs_dict.get("test_greedy_signal"),
            6: obs_dict.get("test_dp_signal"),
            7: obs_dict.get("test_graph_signal"),
            8: obs_dict.get("test_binary_search_signal"),
            9: obs_dict.get("test_math_signal"),
            10: obs_dict.get("test_string_signal"),
        }
        positive = [a for a, v in signals.items() if v is True]
        tested   = [a for a, v in signals.items() if v is not None]

        # If we have a positive signal, submit based on N+TL heuristic
        if positive and obs_dict.get("attempts_left", 3) > 0:
            if n == "large" and tl == "tight":
                return 12  # nlogn
            elif n == "small":
                return 13  # quadratic fine for small
            else:
                return 12  # nlogn safe default

        # Test algorithms we haven't tested yet (prioritise by N/TL match)
        untested = [a for a in self.REASONING_ACTIONS if a not in tested and signals.get(a) is None]
        if untested and info_taken < 4:
            # Prioritise based on N scale
            if n == "large" and tl == "tight":
                priority = [7, 8, 5, 6, 9, 10]  # graph, bs, greedy first
            elif n == "small":
                priority = [6, 9, 10, 5, 7, 8]  # dp, math first
            else:
                priority = [5, 6, 7, 8, 9, 10]
            for a in priority:
                if a in untested:
                    return a

        # Default: submit nlogn (safest mid-range)
        return 12

    # ── Q update ─────────────────────────────────────────────────────────────

    def update(
        self,
        obs_dict: Dict[str, Any],
        action: int,
        reward: float,
        next_obs_dict: Dict[str, Any],
        done: bool,
    ):
        state      = self._encode_state(obs_dict)
        next_state = self._encode_state(next_obs_dict)

        old_q = self.q_table[state][action]
        if done:
            target = reward
        else:
            next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
            target = reward + self.gamma * next_q

        self.q_table[state][action] = old_q + self.alpha * (target - old_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_count += 1

    # ── Serialise for saving ──────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            "epsilon":       self.epsilon,
            "episode_count": self.episode_count,
            "q_table_size":  len(self.q_table),
        }


# ─── Parse observation message into dict ──────────────────────────────────────

def parse_obs(obs) -> Dict[str, Any]:
    """Extract fields from CpArenaEnvObservation into a plain dict."""
    return {
        "revealed_n":              getattr(obs, "revealed_n", None),
        "revealed_time_limit":     getattr(obs, "revealed_time_limit", None),
        "revealed_memory":         getattr(obs, "revealed_memory", None),
        "last_verdict":            getattr(obs, "last_verdict", "none"),
        "attempts_left":           getattr(obs, "attempts_left", 3),
        "info_actions_taken":      getattr(obs, "info_actions_taken", 0),
        "test_greedy_signal":      getattr(obs, "test_greedy_signal", None),
        "test_dp_signal":          getattr(obs, "test_dp_signal", None),
        "test_graph_signal":       getattr(obs, "test_graph_signal", None),
        "test_binary_search_signal": getattr(obs, "test_binary_search_signal", None),
        "test_math_signal":        getattr(obs, "test_math_signal", None),
        "test_string_signal":      getattr(obs, "test_string_signal", None),
        "message":                 getattr(obs, "message", ""),
    }


# ─── LLM Agent (optional) ─────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI competitive programmer solving CP problems.
    Choose action IDs (integers 0-13).
    Actions:
    0: reveal_N  1: reveal_time_limit  2: reveal_memory  3: reveal_tags
    4: reveal_example  5: test_greedy  6: test_dp  7: test_graph
    8: test_binary_search  9: test_math  10: test_string
    11: submit_linear  12: submit_nlogn  13: submit_quadratic
    Strategy: reveal N and time_limit first, then test algorithms, then submit.
    Reply with ONLY a single integer (0-13).
""").strip()

def _heuristic_fallback(step: int, last_message: str) -> int:
    """
    Structured fallback used when LLM is unavailable (no HF_TOKEN).
    Mimics human CP strategy: reveal N → reveal time → test → submit.
    """
    msg = last_message.lower()
    # Phase 1: reveal constraints if not yet known
    if "n scale" not in msg and "large" not in msg and "small" not in msg:
        return 0  # reveal_N
    if "time pressure" not in msg and "tight" not in msg and "loose" not in msg:
        return 1  # reveal_time_limit
    # Phase 2: test algorithms in priority order
    if "plausible" not in msg and "unlikely" not in msg:
        # Pick test based on step to spread coverage
        tests = [7, 5, 6, 8, 9, 10]  # graph, greedy, dp, bs, math, string
        return tests[min(step - 1, len(tests) - 1)]
    # Phase 3: if a plausible signal found, submit nlogn (safest)
    if "plausible" in msg:
        return 12  # submit_nlogn
    # Default: submit nlogn
    return 12


def get_llm_action(client, step: int, last_message: str, history: List[str]) -> int:
    # If no API key, skip LLM entirely and use heuristic
    if not API_KEY:
        print(f"[DEBUG] No API key — using heuristic fallback", flush=True)
        return _heuristic_fallback(step, last_message)

    history_block = "\n".join(history[-10:]) if history else "None"
    user_prompt = f"Step: {step}\nLast observation: {last_message}\nHistory:\n{history_block}\nChoose action (0-13):"
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=10,
            )
            text = (completion.choices[0].message.content or "").strip()
            action_id = int(text.strip().split()[0])
            if 0 <= action_id <= 13:
                return action_id
        except Exception as e:
            print(f"[DEBUG] LLM attempt {attempt+1} failed: {e}", flush=True)

    # All LLM attempts failed — fall back to heuristic instead of always returning 0
    print(f"[DEBUG] LLM failed all attempts — using heuristic fallback", flush=True)
    return _heuristic_fallback(step, last_message)


# ─── Q-learning Training Loop (offline, no Docker needed) ─────────────────────

def train_qlearning(episodes: int = 100):
    """
    Offline Q-learning training using the HTTP server (must be running locally).
    Falls back to a simulation loop if the server is unavailable.
    """
    import urllib.request
    import urllib.error

    base_url = os.getenv("BASE_URL", "http://localhost:8000")

    # Check if server is up
    try:
        urllib.request.urlopen(f"{base_url}/health", timeout=3)
        server_available = True
        print(f"[TRAIN] Server found at {base_url}", flush=True)
    except Exception:
        server_available = False
        print("[TRAIN] Server not available — running simulated training loop", flush=True)

    agents = {task: QLearningAgent() for task in TASKS}
    training_log = {task: [] for task in TASKS}

    for episode in range(1, episodes + 1):
        for task in TASKS:
            agent = agents[task]
            ep_reward = 0.0
            success   = False

            # Emit [START] block — required by validator
            log_start(task=task)

            if server_available:
                ep_reward, success, step_rewards = _run_http_episode(base_url, task, agent)
            else:
                ep_reward, success, step_rewards = _run_simulated_episode(task, agent)

            # Emit [STEP] blocks for each step
            for i, r in enumerate(step_rewards, start=1):
                log_step(step=i, action=i, reward=r, done=(i == len(step_rewards)))

            # Compute score and emit [END] block
            score = max(0.001, min(0.999, (ep_reward + 100) / 300))
            log_end(task=task, success=success, steps=len(step_rewards), score=score, rewards=step_rewards)

            training_log[task].append({
                "episode": episode,
                "reward":  round(ep_reward, 3),
                "success": success,
                "epsilon": round(agent.epsilon, 4),
            })
            agent.decay_epsilon()

        if episode % 10 == 0 or episode == 1:
            summaries = {
                t: f"{sum(r['reward'] for r in training_log[t][-10:]) / min(10, episode):.2f}"
                for t in TASKS
            }
            print(
                f"[TRAIN] Episode {episode}/{episodes} | "
                + " | ".join(f"{t}: avg_r={v}" for t, v in summaries.items()),
                flush=True,
            )

    _save_results(training_log, agents)
    return training_log, agents


def _run_http_episode(base_url: str, task: str, agent: QLearningAgent):
    """Run one episode via the live HTTP server."""
    import urllib.request
    import urllib.error

    def post(endpoint, data):
        req = urllib.request.Request(
            f"{base_url}{endpoint}",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    try:
        reset_resp = post("/reset", {"task": task})
        obs_dict = reset_resp.get("observation", {})
        ep_reward   = 0.0
        step_rewards = []
        prev_obs  = obs_dict

        for step in range(MAX_STEPS):
            action = agent.select_action(obs_dict, task)
            step_resp = post("/step", {"action_id": action})
            reward    = float(step_resp.get("reward", 0.0))
            done      = bool(step_resp.get("done", False))
            next_obs  = step_resp.get("observation", {})

            agent.update(prev_obs, action, reward, next_obs, done)
            ep_reward += reward
            step_rewards.append(reward)
            prev_obs   = next_obs
            obs_dict   = next_obs

            if done:
                break

        score = max(0.0, min(1.0, (ep_reward + 100) / 300))
        success = score >= 0.5
        return ep_reward, success, step_rewards

    except Exception as e:
        print(f"[TRAIN] HTTP episode error ({task}): {e}", flush=True)
        return 0.0, False, []


def _run_simulated_episode(task: str, agent: QLearningAgent):
    """
    Lightweight simulation when the server is unavailable.
    Mimics the reward structure of cp_arena_env_environment.py so Q-values
    are meaningful when deployed against the real server.
    """
    # Synthetic problem
    p = {
        "N_scale":          random.choice(["small", "medium", "large"]),
        "time_pressure":    random.choice(["loose", "normal", "tight"]),
        "memory_pressure":  random.choice(["low", "normal", "high"]),
        "valid_algorithms": random.sample(["greedy", "dp", "graph", "binary_search", "math", "string"], k=random.randint(1, 2)),
        "valid_complexities": random.sample(["linear", "nlogn", "quadratic"], k=random.randint(1, 2)),
    }

    ALGO_MAP      = {5: "greedy", 6: "dp", 7: "graph", 8: "binary_search", 9: "math", 10: "string"}
    COMPLEXITY_MAP = {11: "linear", 12: "nlogn", 13: "quadratic"}
    INFO_COST     = [2, 4, 8, 16, 32]

    obs_dict = {
        "revealed_n": None, "revealed_time_limit": None, "revealed_memory": None,
        "last_verdict": "none", "attempts_left": 3, "info_actions_taken": 0,
        "test_greedy_signal": None, "test_dp_signal": None, "test_graph_signal": None,
        "test_binary_search_signal": None, "test_math_signal": None, "test_string_signal": None,
    }

    SIGNAL_KEYS = {
        5: "test_greedy_signal", 6: "test_dp_signal", 7: "test_graph_signal",
        8: "test_binary_search_signal", 9: "test_math_signal", 10: "test_string_signal",
    }

    used_info   = set()
    last_algo   = None
    ep_reward    = 0.0
    step_rewards = []
    time_left    = 300
    done         = False

    for step in range(MAX_STEPS):
        action = agent.select_action(obs_dict, task)
        reward = 0.0

        # INFO actions
        if action in range(5):
            if action in used_info:
                reward = -5.0
            else:
                cost = INFO_COST[min(obs_dict["info_actions_taken"], 4)]
                reward = -cost
                obs_dict["info_actions_taken"] += 1
                used_info.add(action)
                if action == 0: obs_dict["revealed_n"] = p["N_scale"]
                elif action == 1: obs_dict["revealed_time_limit"] = p["time_pressure"]
                elif action == 2: obs_dict["revealed_memory"] = p["memory_pressure"]
            time_left -= 10

        # REASONING actions
        elif action in range(5, 11):
            algo   = ALGO_MAP[action]
            signal = algo in p["valid_algorithms"]
            reward = 3.0 if signal else -3.0
            obs_dict[SIGNAL_KEYS[action]] = signal
            last_algo = algo
            time_left -= 5

        # SUBMIT actions
        elif action in range(11, 14):
            if obs_dict["attempts_left"] <= 0:
                reward = -50.0
                done   = True
            else:
                complexity = COMPLEXITY_MAP[action]
                obs_dict["attempts_left"] -= 1
                reward -= 10.0  # submission cost

                algo_ok  = last_algo is not None and last_algo in p["valid_algorithms"]
                compl_ok = complexity in p["valid_complexities"]

                if task == "algorithm_selection":
                    if algo_ok:
                        time_bonus = (time_left / 300) * 30
                        reward += 100.0 + time_bonus
                        obs_dict["last_verdict"] = "AC"
                        done = True
                    else:
                        reward += -30.0
                        obs_dict["last_verdict"] = "WA"

                elif task == "complexity_optimization":
                    if compl_ok:
                        eff = max(0, (5 - obs_dict["info_actions_taken"]) * 5)
                        reward += 100.0 + eff
                        obs_dict["last_verdict"] = "AC"
                        done = True
                    else:
                        reward += -20.0
                        obs_dict["last_verdict"] = "TLE"

                elif task == "problem_classification":
                    if algo_ok:
                        reveals = obs_dict["info_actions_taken"]
                        eff     = max(0, 5 - reveals) * 10
                        reward += 60.0 + eff
                        obs_dict["last_verdict"] = "AC"
                        done = True
                    else:
                        reward += -25.0
                        obs_dict["last_verdict"] = "WA"

            time_left -= 20

        if time_left <= 0 and not done:
            reward -= 20.0
            done = True

        next_obs = dict(obs_dict)
        agent.update(obs_dict, action, reward, next_obs, done)
        obs_dict   = next_obs
        ep_reward += reward
        step_rewards.append(reward)

        if done:
            break

    score = max(0.0, min(1.0, (ep_reward + 100) / 300))
    success = score >= 0.5
    return ep_reward, success, step_rewards


def _save_results(training_log, agents):
    """Save training curves and summary to results/."""
    # Full log per task
    for task, log in training_log.items():
        path = os.path.join(RESULTS_DIR, f"training_curve_{task}.json")
        with open(path, "w") as f:
            json.dump(log, f, indent=2)

    # Summary across all tasks
    summary = {}
    for task, log in training_log.items():
        rewards  = [e["reward"] for e in log]
        n        = len(rewards)
        first10  = sum(rewards[:10]) / min(10, n)
        last10   = sum(rewards[-10:]) / min(10, n)
        solved   = sum(1 for e in log if e["success"])
        summary[task] = {
            "episodes":          n,
            "avg_reward_first10": round(first10, 3),
            "avg_reward_last10":  round(last10, 3),
            "improvement":        round(last10 - first10, 3),
            "solve_rate":         round(solved / n, 3),
            "final_epsilon":      round(agents[task].epsilon, 4),
            "q_states_learned":   agents[task].to_dict()["q_table_size"],
        }

    # Random baseline for comparison
    baseline = _compute_random_baseline()
    comparison = {
        "random_policy": baseline,
        "q_learning":    {t: {"avg_reward_last10": v["avg_reward_last10"],
                               "solve_rate":        v["solve_rate"]}
                          for t, v in summary.items()},
    }

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump({"task_summaries": summary, "comparison": comparison}, f, indent=2)

    print(f"\n[RESULTS] Saved to {RESULTS_DIR}/", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("\n[COMPARISON] Random vs Q-learning:", flush=True)
    print(json.dumps(comparison, indent=2), flush=True)


def _compute_random_baseline(episodes: int = 50) -> Dict:
    """Estimate random policy performance for README comparison table."""
    class RandomAgent:
        def select_action(self, obs_dict, task): return random.randint(0, 13)
        def update(self, *args): pass
        def decay_epsilon(self): pass

    results = {}
    for task in TASKS:
        agent = RandomAgent()
        rewards = []
        solved  = 0
        for _ in range(episodes):
            r, s, _ = _run_simulated_episode(task, agent)
            rewards.append(r)
            if s: solved += 1
        results[task] = {
            "avg_reward": round(sum(rewards) / episodes, 3),
            "solve_rate": round(solved / episodes, 3),
        }
    return results


# ─── Async task runner (LLM mode or Docker-based eval) ────────────────────────

async def run_task_llm(task_name: str):
    rewards: List[float] = []
    steps_taken = 0
    score = 0.001
    success = False
    history: List[str] = []

    # log_start BEFORE imports — guarantees [START] appears even if import fails
    log_start(task=task_name)

    try:
        from openai import OpenAI
        from cp_arena_env import CpArenaEnv, CpArenaEnvAction  # type: ignore

        # api_key=API_KEY or "dummy" — OpenAI client raises if key is None
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")

        # Validator runs inference.py inside Docker where server is at localhost:8000
        base_url = os.getenv("BASE_URL", "http://localhost:8000")

        async with CpArenaEnv(base_url=base_url) as env:
            result = await asyncio.wait_for(
                env.reset(task=task_name), timeout=30.0
            )
            last_message = result.observation.message

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                action_id = get_llm_action(client, step, last_message, history)

                try:
                    result = await asyncio.wait_for(
                        env.step(CpArenaEnvAction(action_id=action_id)),
                        timeout=30.0,
                    )
                except Exception as e:
                    print(f"[ERROR] step failed at {step}: {e}", flush=True)
                    break

                obs    = result.observation
                reward = result.reward or 0.0
                done   = result.done
                rewards.append(reward)
                steps_taken  = step
                last_message = obs.message
                log_step(step=step, action=action_id, reward=reward, done=done)
                history.append(
                    f"Step {step}: action={action_id} reward={reward:+.2f} msg={obs.message[:80]}"
                )
                if done:
                    break

        if rewards:
            total_reward = sum(rewards)
            score = max(0.001, min(0.999, (total_reward + 100) / 300))
        success = score >= 0.5

    except Exception as e:
        print(f"[ERROR] run_task crashed: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()

    finally:
        log_end(task=task_name, success=success, steps=steps_taken, score=score, rewards=rewards)


async def main_llm():
    for task in TASKS:
        await run_task_llm(task)
        await asyncio.sleep(2)


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if os.getenv("TRAIN_QL", "0") == "1":
        # Explicit opt-in: TRAIN_QL=1 python inference.py
        print(f"[MODE] Q-learning offline training — {EPISODES} episodes per task", flush=True)
        train_qlearning(episodes=EPISODES)
    else:
        # Default: LLM agent — this is what the validator runs
        # Works with or without HF_TOKEN (falls back to heuristic if no key)
        asyncio.run(main_llm())
