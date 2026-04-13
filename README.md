

# CP-Arena: Decision-Making Under Uncertainty — An RL Environment for Algorithm Reasoning

CP-Arena is a **partially observable reinforcement learning environment** that models one of the hardest real-world cognitive skills: making correct decisions with incomplete information, under time pressure, with costly mistakes.

The agent faces a hidden problem. It must decide what to reveal, what to test, and when to commit — just like an engineer triaging a production incident, a doctor narrowing a diagnosis, or a trader reading a market. The domain is algorithm selection; the skill being learned is universal.

> A multi-task RL environment where each episode evaluates a distinct decision-making objective in competitive programming — algorithm selection, complexity feasibility, or problem classification.

> CP-Arena does not teach an agent to write code. It teaches an agent how to reason and decide — which information is worth gathering, which hypotheses to test, and when to commit to a solution.

---

## Task Structure

CP-Arena operates as a multi-task reinforcement learning environment. Each episode runs in exactly **one of three distinct task modes**, each with its own objective, grading logic, and success criteria.

| Task | Objective | Key Signal | Graded On |
| --- | --- | --- | --- |
| `algorithm_selection` | Identify the correct algorithmic paradigm | `test_*_signal` responses | Correctness of chosen paradigm |
| `complexity_optimization` | Select a feasible time complexity | N scale + time pressure | Feasibility under constraints |
| `problem_classification` | Classify with minimum information reveals | Tag inference from partial obs | Accuracy + efficiency (fewer reveals = higher score) |

Tasks are **mutually exclusive per episode** — the environment dynamically adjusts its reward function, available signals, and success criteria based on the active task. Each task is graded independently, ensuring clear separation of objectives.

### Task 1 — Algorithm Selection
The agent must identify which algorithmic paradigm solves the hidden problem. Complexity is not graded — only whether the agent chose the correct algorithm family (Graph, DP, Greedy, etc.).

### Task 2 — Complexity Optimization
The algorithm is secondary. The agent must commit to a time complexity that is feasible given the revealed constraints (N scale, time limit). Choosing O(N²) on a large-N tight-time problem = TLE penalty.

### Task 3 — Problem Classification
The agent is rewarded for correctness **and** efficiency. Fewer information reveals = higher efficiency score. This task specifically tests whether the agent learns to reason from minimal observations — the core RL challenge.

---

## Submission Checklist

| Requirement | Status |
| --- | --- |
| HF Space deployed and running | ✅ |
| OpenEnv spec compliant (openenv.yaml) | ✅ |
| Dockerfile builds and runs | ✅ |
| 3 tasks with graders (scores 0.0–1.0) | ✅ |
| inference.py with correct log format | ✅ |
| Offline execution — no external APIs | ✅ |
| Docker-first execution | ✅ |
| validate-submission.sh passes all 3 checks | ✅ |
| Native Q-learning agent (no API key needed) | ✅ |
| Training curves saved to results/ | ✅ |
| /leaderboard endpoint with per-task stats | ✅ |
| Natural language problem hints (action 14) | ✅ |

---

## Mandatory File Tree

```
cp_arena_env/
├── inference.py                     # Q-learning agent + optional LLM agent
├── Dockerfile                       # Docker build (root level)
├── openenv.yaml                     # OpenEnv spec
├── requirements.txt                 # Python dependencies
├── validate-submission.sh           # Pre-submission validator
├── README.md                        # This file
├── __init__.py                      # Package exports
├── client.py                        # Environment client interface
├── models.py                        # Action/Observation/State models
├── pyproject.toml                   # Project metadata
├── results/                         # Auto-generated training output
│   ├── training_curve_algorithm_selection.json
│   ├── training_curve_complexity_optimization.json
│   ├── training_curve_problem_classification.json
│   └── summary.json
└── server/
    ├── app.py                       # FastAPI server + leaderboard endpoints
    ├── cp_arena_env_environment.py  # Core environment logic
    └── dataset/
        └── problems.json            # 53 CP problems (local, offline)
```

---

## Quick Start

### 1. Install

```bash
pip install openenv-core
uv sync
```

### 2. Run locally

```bash
uv run uvicorn server.app:app --reload --port 8000
```

### 3. Run with Docker

```bash
docker build -t cp-arena-env:latest .
docker run -p 8000:8000 cp-arena-env:latest
```

### 4. Run inference (offline baseline agent — no token needed)

```bash
python inference.py
```

Runs an offline Q-learning agent for 100 episodes per task. Results are saved automatically:

```
results/training_curve_algorithm_selection.json
results/training_curve_complexity_optimization.json
results/training_curve_problem_classification.json
results/summary.json    # Random baseline vs Q-learning comparison
```

To run Q-learning training instead (offline, no token needed):

```bash
TRAIN_QL=1 python inference.py

# Custom episode count
TRAIN_QL=1 EPISODES=200 python inference.py
```

The LLM agent is the **default** — it runs automatically when the validator executes `inference.py`. It uses `HF_TOKEN` if set, and falls back to a structured heuristic policy if not.

> `inference.py` is a demo script only. It is not required for environment correctness. The environment runs fully without it.

### 5. Validate before submitting

```bash
bash validate-submission.sh <YOUR_HF_SPACE_URL>
# Example: bash validate-submission.sh https://deekshitha08-cp-arena-env.hf.space
```

Expected output:

```
Step 1/3: Pinging HF Space ...
PASSED -- HF Space is live and responds to /reset
Step 2/3: Running docker build ...
PASSED -- Docker build succeeded
Step 3/3: Running openenv validate ...
PASSED -- openenv validate passed
  [OK] cp_arena: Ready for multi-mode deployment

All 3/3 checks passed!
Your submission is ready to submit.
```

---

## Offline Execution — Compliance Statement

CP-Arena is fully self-contained and runs completely offline:

* No external APIs required at runtime
* No cloud database dependencies
* No internet access needed inside the container
* Dataset stored locally at `server/dataset/problems.json`
* Deterministic grading logic — same input always gives same output
* Docker-first execution — fully reproducible inside container
* OpenEnv-compatible structure — passes `openenv validate`
* Q-learning agent in `inference.py` trains with zero external dependencies

---

## Quick Evaluation Example

Problem: cf\_031 | Difficulty: 1600

| Step | Action | Observation | Reward |
| --- | --- | --- | --- |
| 1 | reveal\_N | N scale: large | -2.0 |
| 2 | reveal\_time\_limit | Time: tight | -4.0 |
| 3 | test\_graph | PLAUSIBLE | +3.0 |
| 4 | reveal\_statement\_hint | Natural language hint revealed | -5.0 |
| 5 | submit\_nlogn | AC | +117.5 |

**Total reward: 109.5 | Score: 0.698 | Steps: 5**

---

## Three Tasks

| Task | Objective | Success Metric |
| --- | --- | --- |
| algorithm\_selection | Identify correct algorithm | Correct paradigm chosen |
| complexity\_optimization | Choose optimal time complexity | Feasibility under constraints |
| problem\_classification | Classify with minimum reveals | Efficiency + correctness score |

### Task 1 — Algorithm Selection

Agent reveals constraints, tests algorithm hypotheses, and submits. Grader checks whether the chosen algorithm is valid.

* Correct algorithm: +100 + time bonus (up to +30)
* Wrong algorithm: -30
* Submission cost: -10 per attempt
* Info gathering: exponential cost (-2, -4, -8, -16, -32)

### Task 2 — Complexity Optimization

Agent commits to a time complexity. Grader checks feasibility under constraints.

* Correct complexity: +100 + efficiency bonus
* Better than needed: +50 (partial credit)
* TLE: -20

### Task 3 — Problem Classification

Agent is rewarded for correctness AND efficiency. Fewer reveals = higher score.

* Correct classification: +60 + efficiency score (up to +50)
* Efficiency score = (5 - reveals\_used) × 10
* Wrong classification: -25

---

## Normalized Score

```
score = (total_reward + 100) / 300
score = clamp(score, 0.001, 0.999)
```

## Structured Log Format

`inference.py` emits structured stdout blocks parsed by the validator:

```
[START] task=algorithm_selection env=cp_arena_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=0 reward=-2.00 done=false error=null
[STEP] step=2 action=1 reward=-4.00 done=false error=null
[STEP] step=3 action=7 reward=3.00 done=false error=null
[STEP] step=4 action=12 reward=117.50 done=true error=null
[END] success=true steps=4 score=0.72 rewards=-2.00,-4.00,3.00,117.50
```

---

## Action Space (15 Actions)

| ID | Action | Type | Cost / Signal | Description |
| --- | --- | --- | --- | --- |
| 0 | reveal\_N | Info | exponential | Reveal input size scale |
| 1 | reveal\_time\_limit | Info | exponential | Reveal time pressure |
| 2 | reveal\_memory | Info | exponential | Reveal memory pressure |
| 3 | reveal\_tags | Info | exponential | Reveal problem category tags |
| 4 | reveal\_example | Info | exponential | Reveal sample I/O pattern |
| 5 | test\_greedy | Reasoning | +3 / -3 | Test if greedy is plausible |
| 6 | test\_dp | Reasoning | +3 / -3 | Test if DP is plausible |
| 7 | test\_graph | Reasoning | +3 / -3 | Test if graph approach is plausible |
| 8 | test\_binary\_search | Reasoning | +3 / -3 | Test if binary search is plausible |
| 9 | test\_math | Reasoning | +3 / -3 | Test if math approach is plausible |
| 10 | test\_string | Reasoning | +3 / -3 | Test if string approach is plausible |
| 11 | submit\_linear | Submit | -10 | Submit O(N) solution |
| 12 | submit\_nlogn | Submit | -10 | Submit O(N log N) solution |
| 13 | submit\_quadratic | Submit | -10 | Submit O(N²) solution |
| 14 | reveal\_statement\_hint | Info | -5 | Reveal natural language problem description |

---

## Observation Space

```json
{
  "revealed_n": "large",
  "revealed_time_limit": "tight",
  "revealed_memory": "normal",
  "revealed_tags": null,
  "revealed_example": false,
  "revealed_statement_hint": null,
  "test_greedy_signal": null,
  "test_dp_signal": null,
  "test_graph_signal": true,
  "test_binary_search_signal": null,
  "test_math_signal": null,
  "test_string_signal": null,
  "time_remaining": 245,
  "attempts_left": 2,
  "last_verdict": "none",
  "last_reward": 3.0,
  "step_count": 3,
  "info_actions_taken": 2,
  "message": "test_graph: PLAUSIBLE (+3)"
}
```

---

## Reward Design

| Event | Reward |
| --- | --- |
| AC — Algorithm Selection | +100 + time bonus |
| AC — Complexity Optimization | +100 + efficiency bonus |
| AC — Problem Classification | +60 + efficiency bonus |
| WA wrong algorithm | -30 |
| TLE wrong complexity | -20 |
| All attempts exhausted | -50 |
| Time runs out | -20 |
| Info reveal (exponential) | -2 / -4 / -8 / -16 / -32 |
| Reasoning action | +3 or -3 |
| Submission attempt | -10 |
| Repeated reveal | -5 |
| Inconsistent complexity | -5 |
| reveal\_statement\_hint | -5 |

---

## Q-Learning Agent — Training Results

The Q-learning agent (`inference.py`) trains fully offline — no API key, no Docker required.
It uses tabular Q-learning (α=0.1, γ=0.95, ε-decay=0.97) with structured ε-greedy exploration
that mimics human CP strategy: reveal N → reveal time limit → test algorithms → submit.

Run `python inference.py` to reproduce these results locally.

### Random Policy vs Trained Q-Learning Agent

| Task | Random Avg Reward | Q-Learning Avg Reward (last 10 eps) | Random Solve Rate | Q-Learning Solve Rate |
| --- | --- | --- | --- | --- |
| algorithm\_selection | ~-15 | ~+85 | ~5% | ~70% |
| complexity\_optimization | ~-10 | ~+90 | ~8% | ~75% |
| problem\_classification | ~-20 | ~+65 | ~3% | ~60% |

*Results from 100 training episodes per task. A random agent scores ~0.30 on average; the Q-learning agent reaches ~0.70+ by episode 50 — demonstrating the reward signal produces genuine learning.*

---

## Dataset

53 problems, difficulties 800–2000 across greedy, dp, graph, binary\_search, math, string, brute\_force with complexities O(N), O(N log N), O(N²). Every problem includes a `statement_hint` (natural language description) and an `example_hint` (concrete I/O example) used by action 14.

```json
{
  "id": "cf_031",
  "N_scale": "large",
  "time_pressure": "tight",
  "memory_pressure": "normal",
  "valid_algorithms": ["graph"],
  "valid_complexities": ["nlogn"],
  "tags": ["graph", "topological_sort"],
  "difficulty": 1600,
  "statement_hint": "Order N tasks given prerequisite constraints — detect cycles and produce valid ordering.",
  "example_hint": "Input: 6 tasks with 5 dependency edges. Output: valid topological order or CYCLE."
}
```

---

## Why This Is Genuine RL

| Property | Description |
| --- | --- |
| Partial observability | Agent starts with zero information every episode |
| Sequential decisions | Earlier actions affect later states and available information |
| Exploration needed | Agent must learn which reveals are most diagnostic per task |
| Shaped reward | Multi-component reward encourages efficient, correct strategy |
| Belief updating | Reasoning verdicts update available state for retries |
| Multiple tasks | 3 distinct evaluation objectives with different optimal policies |
| Leaderboard tracking | Every completed episode recorded; top-50 per task maintained in memory |

---

## API Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| /reset | POST | Start new episode. Body: `{"task": "algorithm_selection"}` |
| /step | POST | Take action. Body: `{"action_id": 0}` |
| /state | GET | Get current episode metadata |
| /health | GET | Health check — returns 200 if live |
| /leaderboard | GET | All tasks — top runs, solve rate, avg reward per task |
| /leaderboard/{task} | GET | Single task leaderboard |
| /stats | GET | Flat summary stats across all tasks |

---

## Python Client Example

```python
import asyncio
from cp_arena_env import CpArenaEnvAction, CpArenaEnv

async def main():
    client = await CpArenaEnv.from_env("Deekshitha08/cp_arena_env")
    async with client:
        result = await client.reset(task="algorithm_selection")
        print(result.observation.message)

        result = await client.step(CpArenaEnvAction(action_id=0))   # reveal_N
        result = await client.step(CpArenaEnvAction(action_id=14))  # reveal_statement_hint
        result = await client.step(CpArenaEnvAction(action_id=7))   # test_graph
        result = await client.step(CpArenaEnvAction(action_id=12))  # submit_nlogn
        print(result.observation.last_verdict)
        print(result.reward)

asyncio.run(main())
```

---

## Tech Stack

OpenEnv · FastAPI · Uvicorn · Pydantic · Docker · Hugging Face Spaces
