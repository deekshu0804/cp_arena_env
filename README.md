---
title: CP Arena Env
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
base_path: /health
pinned: false
---
# CP-Arena: Decision-Making Under Uncertainty — An RL Environment for Algorithm Reasoning

CP-Arena is a **partially observable reinforcement learning environment** that models one of the hardest real-world cognitive skills: **making correct decisions with incomplete information, under time pressure, with costly mistakes**.

The agent faces a hidden problem. It must decide what to reveal, what to test, and when to commit — just like an engineer triaging a production incident, a doctor narrowing a diagnosis, or a trader reading a market. The domain is algorithm selection; the skill being learned is universal.

> CP-Arena does not teach an agent to write code. It teaches an agent **how to reason and decide** — which information is worth gathering, which hypotheses to test, and when to commit to a solution.

---

## Quick Evaluation Example

Problem: cf_031 | Difficulty: 1600

| Step | Action | Observation | Reward |
|---|---|---|---|
| 1 | reveal_N | N scale: large | -2.0 |
| 2 | reveal_time_limit | Time: tight | -4.0 |
| 3 | test_graph | PLAUSIBLE | +3.0 |
| 4 | submit_nlogn | AC | +117.5 |

**Total reward: 114.5 | Score: 0.715 | Steps: 4**

Final evaluation:
- Algorithm Selection: PASS
- Complexity Feasibility: PASS
- Information Efficiency: GOOD (solved in 4 steps)

---

## Task Evaluation Summary

| Task | Objective | Success Metric |
|---|---|---|
| algorithm_selection | Identify correct algorithm | Correct paradigm chosen |
| complexity_optimization | Choose valid complexity | Feasibility under constraints |
| problem_classification | Classify with minimum reveals | Efficiency score |

---

## Repository Structure

```
cp_arena_env/
├── __init__.py                      # Package exports
├── client.py                        # Environment client interface
├── models.py                        # Action/Observation/State models
├── inference.py                     # Demo agent runner (LLM-powered)
├── Dockerfile                       # Container setup
├── openenv.yaml                     # OpenEnv spec
├── pyproject.toml                   # Dependencies
├── requirements.txt                 # Python requirements
├── README.md                        # This file
├── validate-submission.sh           # Pre-submission validator
└── server/
    ├── app.py                       # FastAPI server
    ├── cp_arena_env_environment.py  # Core environment logic
    └── dataset/
        └── problems.json            # 50 CP problems dataset
```

---

## Local Setup

```bash
# Install dependencies
pip install openenv-core
uv sync

# Start environment server locally
uv run uvicorn server.app:app --reload --port 8000

# Open web interface
# http://localhost:8000/web

# Run demo agent
uv run inference.py

# Build Docker container
docker build -t cp-arena-env:latest .
docker run -p 8000:8000 cp-arena-env:latest
```

### Environment Variables

Set these before running `inference.py`:

```bash
export HF_TOKEN=your_huggingface_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export IMAGE_NAME=cp-arena-env:latest

python inference.py
```

---

## Pre-Submission Validation

Run the validator before submitting:

```bash
bash validate-submission.sh
```

Expected output:

```
[06:56:33] Repo:     /c/Users/v deekshitha/cp_arena_env
[06:56:33] Ping URL: https://deekshitha08-cp-arena-env.hf.space

[06:56:33] Step 1/3: Pinging HF Space ...
[06:56:35] PASSED -- HF Space is live and responds to /reset
[06:56:35] Step 2/3: Running docker build ...
[06:56:35]   Found Dockerfile in /c/Users/v deekshitha/cp_arena_env
[06:58:13] PASSED -- Docker build succeeded
[06:58:13] Step 3/3: Running openenv validate ...
[06:58:14] PASSED -- openenv validate passed
[06:58:14]   [OK] cp_arena: Ready for multi-mode deployment

All 3/3 checks passed!
Your submission is ready to submit.
```

---

## Offline Execution Guarantee

CP-Arena is fully self-contained:

- No external APIs required
- No cloud database dependencies
- No internet dependency at runtime
- Dataset stored locally (problems.json)
- Deterministic grading logic
- Fully reproducible inside Docker container

All evaluation runs fully offline inside the container.

---

## Submission Artifacts

This submission includes:

- OpenEnv compatible environment
- 3 defined tasks with separate grading logic
- Dataset (50 CP problems, difficulties 800-2000)
- LLM-powered inference script (inference.py)
- Docker container (Dockerfile in root)
- Public GitHub repository
- HuggingFace Spaces deployment
- Pre-submission validation script
- Full documentation

---

## What This Environment Does

Most RL environments give agents complete information upfront. CP-Arena does not.

The agent starts blind. Every episode, a hidden problem is loaded. The agent must spend actions to reveal information, test reasoning hypotheses about which algorithm fits, commit to a solution, and get a verdict — all under time pressure.

This is a **Partially Observable MDP** — the agent must learn which information is most diagnostic without revealing everything. The same decision structure appears in medical diagnosis, financial analysis, and engineering triage. CP-Arena provides a clean, verifiable, fully automated version of this challenge.

---

## Three Task Definitions

CP-Arena supports three distinct evaluation tasks, each with its own objective, success condition, and grading logic.

### Task 1 — Algorithm Selection

**Objective:** Agent must identify the correct algorithm class for a given problem.

**How it works:** Agent reveals constraints, tests algorithm hypotheses, and submits a solution approach. The grader checks whether the chosen algorithm is valid for the problem.

**Success condition:** Correct algorithm submitted within attempt limit.

**Grading logic:**
- Correct algorithm: +100 + time bonus (up to +30)
- Wrong algorithm: -30
- Submission cost: -10 per attempt
- Info gathering: exponential cost (-2, -4, -8, -16, -32)

---

### Task 2 — Complexity Optimization

**Objective:** Agent must select the optimal time complexity under the given constraints.

**How it works:** Agent analyzes problem size and time pressure, then commits to a complexity class. The grader checks whether the chosen complexity is feasible.

**Success condition:** Correct complexity submitted without TLE.

**Grading logic:**
- Correct complexity: +100 + efficiency bonus
- Better than needed: +50 (partial credit)
- TLE wrong complexity: -20
- Efficiency bonus based on minimal info usage

---

### Task 3 — Problem Classification

**Objective:** Agent must correctly classify a problem using the minimum number of information reveals.

**How it works:** Agent is rewarded not just for correctness but for efficiency. Solving with fewer reveals scores higher.

**Success condition:** Correct classification with minimal information cost.

**Grading logic:**
- Correct classification: +60 + efficiency score (up to +50)
- Efficiency score = (5 - reveals_used) x 10
- Wrong classification: -25

---

## Normalized Score

All tasks return a normalized score in [0.0, 1.0]:

```
score = (total_reward + 100) / 300
score = clamp(score, 0.0, 1.0)
```

---

## Action Space (14 Actions)

| ID | Action | Type | Description |
|---|---|---|---|
| 0 | reveal_N | Info | Reveal input size scale |
| 1 | reveal_time_limit | Info | Reveal time pressure |
| 2 | reveal_memory | Info | Reveal memory pressure |
| 3 | reveal_tags | Info | Reveal problem category tags |
| 4 | reveal_example | Info | Reveal sample I/O pattern |
| 5 | test_greedy | Reasoning | Test if greedy is plausible |
| 6 | test_dp | Reasoning | Test if DP is plausible |
| 7 | test_graph | Reasoning | Test if graph approach is plausible |
| 8 | test_binary_search | Reasoning | Test if binary search is plausible |
| 9 | test_math | Reasoning | Test if math approach is plausible |
| 10 | test_string | Reasoning | Test if string approach is plausible |
| 11 | submit_linear | Submit | Submit O(N) solution |
| 12 | submit_nlogn | Submit | Submit O(N log N) solution |
| 13 | submit_quadratic | Submit | Submit O(N^2) solution |

---

## Observation Space

```json
{
  "revealed_n": "large",
  "revealed_time_limit": "tight",
  "revealed_memory": "normal",
  "revealed_tags": null,
  "revealed_example": false,
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

### Core Rewards

| Event | Reward |
|---|---|
| AC Algorithm Selection | +100 + time bonus |
| AC Complexity Optimization | +100 + efficiency bonus |
| AC Problem Classification | +60 + efficiency bonus |
| WA wrong algorithm | -30 |
| TLE wrong complexity | -20 |
| All attempts exhausted | -50 |
| Time runs out | -20 |

### Action Costs

| Action Type | Cost |
|---|---|
| Info reveal 1st | -2 |
| Info reveal 2nd | -4 |
| Info reveal 3rd | -8 |
| Info reveal 4th | -16 |
| Info reveal 5th | -32 |
| Reasoning action | +3 or -3 |
| Submission attempt | -10 |
| Repeated reveal | -5 |
| Inconsistent complexity | -5 |

---

## Dataset

50 competitive programming problems spanning difficulties 800 to 2000, covering algorithm types greedy, dp, graph, binary_search, math, string, and brute_force, with complexities O(N), O(N log N), O(N^2), across small, medium, and large input sizes.

```json
{
  "id": "cf_031",
  "N_scale": "large",
  "time_pressure": "tight",
  "memory_pressure": "normal",
  "valid_algorithms": ["graph"],
  "valid_complexities": ["nlogn"],
  "tags": ["graph", "topological_sort"],
  "difficulty": 1600
}
```

---

## Why This Is Genuine RL

| Property | Description |
|---|---|
| Partial observability | Agent starts with zero information |
| Sequential decisions | Earlier actions affect later states |
| Exploration needed | Agent must learn which reveals are diagnostic |
| Shaped reward | Multi-component reward encourages strategy |
| Belief updating | Verdicts update available state for retries |
| Multiple tasks | 3 distinct evaluation objectives |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| /reset | POST | Start new episode. Accepts task parameter |
| /step | POST | Take action with action_id 0-13 |
| /state | GET | Get current episode metadata |
| /health | GET | Health check |
| /web | GET | Interactive web UI |

---

## Quick Start

```python
import asyncio
from cp_arena_env import CpArenaAction, CpArenaEnv

async def main():
    client = await CpArenaEnv.from_env("Deekshitha08/cp_arena_env")
    async with client:
        result = await client.reset(task="algorithm_selection")
        print(result.observation.message)

        result = await client.step(CpArenaAction(action_id=0))
        result = await client.step(CpArenaAction(action_id=7))
        result = await client.step(CpArenaAction(action_id=12))
        print(result.observation.last_verdict)
        print(result.reward)

asyncio.run(main())
```

---

## Running All 3 Tasks

```python
TASKS = [
    "algorithm_selection",
    "complexity_optimization",
    "problem_classification"
]

for task in TASKS:
    result = await client.reset(task=task)
```

---

## Environment Design Philosophy

CP-Arena models the decision process of a competitive programmer, not the act of writing code. The agent learns which constraints are most diagnostic, which algorithms fit which patterns, how to balance information gathering vs time efficiency, and how to recover from wrong hypotheses across three progressively nuanced task objectives.

The core skill — **reasoning under partial information with costly actions** — is one of the most transferable capabilities an RL agent can develop. CP-Arena provides a structured, verifiable, fully automated environment to measure and improve it.

---

## Tech Stack

OpenEnv environment framework, FastAPI and Uvicorn for the HTTP server, Pydantic for typed models, Docker for containerized deployment, and Hugging Face Spaces for hosting.

---# CP-Arena: Decision-Making Under Uncertainty — An RL Environment for Algorithm Reasoning

CP-Arena is a **partially observable reinforcement learning environment** that models one of the hardest real-world cognitive skills: **making correct decisions with incomplete information, under time pressure, with costly mistakes**.

The agent faces a hidden problem. It must decide what to reveal, what to test, and when to commit — just like an engineer triaging a production incident, a doctor narrowing a diagnosis, or a trader reading a market. The domain is algorithm selection; the skill being learned is universal.

> CP-Arena does not teach an agent to write code. It teaches an agent **how to reason and decide** — which information is worth gathering, which hypotheses to test, and when to commit to a solution.

---

## Quick Evaluation Example

Problem: cf_031 | Difficulty: 1600

| Step | Action | Observation | Reward |
|---|---|---|---|
| 1 | reveal_N | N scale: large | -2.0 |
| 2 | reveal_time_limit | Time: tight | -4.0 |
| 3 | test_graph | PLAUSIBLE | +3.0 |
| 4 | submit_nlogn | AC | +117.5 |

**Total reward: 114.5 | Score: 0.715 | Steps: 4**

Final evaluation:
- Algorithm Selection: PASS
- Complexity Feasibility: PASS
- Information Efficiency: GOOD (solved in 4 steps)

---

## Task Evaluation Summary

| Task | Objective | Success Metric |
|---|---|---|
| algorithm_selection | Identify correct algorithm | Correct paradigm chosen |
| complexity_optimization | Choose valid complexity | Feasibility under constraints |
| problem_classification | Classify with minimum reveals | Efficiency score |

---

## Repository Structure

```
cp_arena_env/
├── __init__.py                      # Package exports
├── client.py                        # Environment client interface
├── models.py                        # Action/Observation/State models
├── inference.py                     # Demo agent runner (LLM-powered)
├── Dockerfile                       # Container setup
├── openenv.yaml                     # OpenEnv spec
├── pyproject.toml                   # Dependencies
├── requirements.txt                 # Python requirements
├── README.md                        # This file
├── validate-submission.sh           # Pre-submission validator
└── server/
    ├── app.py                       # FastAPI server
    ├── cp_arena_env_environment.py  # Core environment logic
    └── dataset/
        └── problems.json            # 50 CP problems dataset
```

---

## Local Setup

```bash
# Install dependencies
pip install openenv-core
uv sync

# Start environment server locally
uv run uvicorn server.app:app --reload --port 8000

# Open web interface
# http://localhost:8000/web

# Run demo agent
uv run inference.py

# Build Docker container
docker build -t cp-arena-env:latest .
docker run -p 8000:8000 cp-arena-env:latest
```

### Environment Variables

Set these before running `inference.py`:

```bash
export HF_TOKEN=your_huggingface_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export IMAGE_NAME=cp-arena-env:latest

python inference.py
```

---

## Pre-Submission Validation

Run the validator before submitting:

```bash
bash validate-submission.sh
```

Expected output:

```
[06:56:33] Repo:     /c/Users/v deekshitha/cp_arena_env
[06:56:33] Ping URL: https://deekshitha08-cp-arena-env.hf.space

[06:56:33] Step 1/3: Pinging HF Space ...
[06:56:35] PASSED -- HF Space is live and responds to /reset
[06:56:35] Step 2/3: Running docker build ...
[06:56:35]   Found Dockerfile in /c/Users/v deekshitha/cp_arena_env
[06:58:13] PASSED -- Docker build succeeded
[06:58:13] Step 3/3: Running openenv validate ...
[06:58:14] PASSED -- openenv validate passed
[06:58:14]   [OK] cp_arena: Ready for multi-mode deployment

All 3/3 checks passed!
Your submission is ready to submit.
```

---

## Offline Execution Guarantee

CP-Arena is fully self-contained:

- No external APIs required
- No cloud database dependencies
- No internet dependency at runtime
- Dataset stored locally (problems.json)
- Deterministic grading logic
- Fully reproducible inside Docker container

All evaluation runs fully offline inside the container.

---

## Submission Artifacts

This submission includes:

- OpenEnv compatible environment
- 3 defined tasks with separate grading logic
- Dataset (50 CP problems, difficulties 800-2000)
- LLM-powered inference script (inference.py)
- Docker container (Dockerfile in root)
- Public GitHub repository
- HuggingFace Spaces deployment
- Pre-submission validation script
- Full documentation

---

## What This Environment Does

Most RL environments give agents complete information upfront. CP-Arena does not.

The agent starts blind. Every episode, a hidden problem is loaded. The agent must spend actions to reveal information, test reasoning hypotheses about which algorithm fits, commit to a solution, and get a verdict — all under time pressure.

This is a **Partially Observable MDP** — the agent must learn which information is most diagnostic without revealing everything. The same decision structure appears in medical diagnosis, financial analysis, and engineering triage. CP-Arena provides a clean, verifiable, fully automated version of this challenge.

---

## Three Task Definitions

CP-Arena supports three distinct evaluation tasks, each with its own objective, success condition, and grading logic.

### Task 1 — Algorithm Selection

**Objective:** Agent must identify the correct algorithm class for a given problem.

**How it works:** Agent reveals constraints, tests algorithm hypotheses, and submits a solution approach. The grader checks whether the chosen algorithm is valid for the problem.

**Success condition:** Correct algorithm submitted within attempt limit.

**Grading logic:**
- Correct algorithm: +100 + time bonus (up to +30)
- Wrong algorithm: -30
- Submission cost: -10 per attempt
- Info gathering: exponential cost (-2, -4, -8, -16, -32)

---

### Task 2 — Complexity Optimization

**Objective:** Agent must select the optimal time complexity under the given constraints.

**How it works:** Agent analyzes problem size and time pressure, then commits to a complexity class. The grader checks whether the chosen complexity is feasible.

**Success condition:** Correct complexity submitted without TLE.

**Grading logic:**
- Correct complexity: +100 + efficiency bonus
- Better than needed: +50 (partial credit)
- TLE wrong complexity: -20
- Efficiency bonus based on minimal info usage

---

### Task 3 — Problem Classification

**Objective:** Agent must correctly classify a problem using the minimum number of information reveals.

**How it works:** Agent is rewarded not just for correctness but for efficiency. Solving with fewer reveals scores higher.

**Success condition:** Correct classification with minimal information cost.

**Grading logic:**
- Correct classification: +60 + efficiency score (up to +50)
- Efficiency score = (5 - reveals_used) x 10
- Wrong classification: -25

---

## Normalized Score

All tasks return a normalized score in [0.0, 1.0]:

```
score = (total_reward + 100) / 300
score = clamp(score, 0.0, 1.0)
```

---

## Action Space (14 Actions)

| ID | Action | Type | Description |
|---|---|---|---|
| 0 | reveal_N | Info | Reveal input size scale |
| 1 | reveal_time_limit | Info | Reveal time pressure |
| 2 | reveal_memory | Info | Reveal memory pressure |
| 3 | reveal_tags | Info | Reveal problem category tags |
| 4 | reveal_example | Info | Reveal sample I/O pattern |
| 5 | test_greedy | Reasoning | Test if greedy is plausible |
| 6 | test_dp | Reasoning | Test if DP is plausible |
| 7 | test_graph | Reasoning | Test if graph approach is plausible |
| 8 | test_binary_search | Reasoning | Test if binary search is plausible |
| 9 | test_math | Reasoning | Test if math approach is plausible |
| 10 | test_string | Reasoning | Test if string approach is plausible |
| 11 | submit_linear | Submit | Submit O(N) solution |
| 12 | submit_nlogn | Submit | Submit O(N log N) solution |
| 13 | submit_quadratic | Submit | Submit O(N^2) solution |

---

## Observation Space

```json
{
  "revealed_n": "large",
  "revealed_time_limit": "tight",
  "revealed_memory": "normal",
  "revealed_tags": null,
  "revealed_example": false,
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

### Core Rewards

| Event | Reward |
|---|---|
| AC Algorithm Selection | +100 + time bonus |
| AC Complexity Optimization | +100 + efficiency bonus |
| AC Problem Classification | +60 + efficiency bonus |
| WA wrong algorithm | -30 |
| TLE wrong complexity | -20 |
| All attempts exhausted | -50 |
| Time runs out | -20 |

### Action Costs

| Action Type | Cost |
|---|---|
| Info reveal 1st | -2 |
| Info reveal 2nd | -4 |
| Info reveal 3rd | -8 |
| Info reveal 4th | -16 |
| Info reveal 5th | -32 |
| Reasoning action | +3 or -3 |
| Submission attempt | -10 |
| Repeated reveal | -5 |
| Inconsistent complexity | -5 |

---

## Dataset

50 competitive programming problems spanning difficulties 800 to 2000, covering algorithm types greedy, dp, graph, binary_search, math, string, and brute_force, with complexities O(N), O(N log N), O(N^2), across small, medium, and large input sizes.

```json
{
  "id": "cf_031",
  "N_scale": "large",
  "time_pressure": "tight",
  "memory_pressure": "normal",
  "valid_algorithms": ["graph"],
  "valid_complexities": ["nlogn"],
  "tags": ["graph", "topological_sort"],
  "difficulty": 1600
}
```

---

## Why This Is Genuine RL

| Property | Description |
|---|---|
| Partial observability | Agent starts with zero information |
| Sequential decisions | Earlier actions affect later states |
| Exploration needed | Agent must learn which reveals are diagnostic |
| Shaped reward | Multi-component reward encourages strategy |
| Belief updating | Verdicts update available state for retries |
| Multiple tasks | 3 distinct evaluation objectives |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| /reset | POST | Start new episode. Accepts task parameter |
| /step | POST | Take action with action_id 0-13 |
| /state | GET | Get current episode metadata |
| /health | GET | Health check |
| /web | GET | Interactive web UI |

---

## Quick Start

```python
import asyncio
from cp_arena_env import CpArenaAction, CpArenaEnv

async def main():
    client = await CpArenaEnv.from_env("Deekshitha08/cp_arena_env")
    async with client:
        result = await client.reset(task="algorithm_selection")
        print(result.observation.message)

        result = await client.step(CpArenaAction(action_id=0))
        result = await client.step(CpArenaAction(action_id=7))
        result = await client.step(CpArenaAction(action_id=12))
        print(result.observation.last_verdict)
        print(result.reward)

asyncio.run(main())
```

---

## Running All 3 Tasks

```python
TASKS = [
    "algorithm_selection",
    "complexity_optimization",
    "problem_classification"
]

for task in TASKS:
    result = await client.reset(task=task)
```

---

## Environment Design Philosophy

CP-Arena models the decision process of a competitive programmer, not the act of writing code. The agent learns which constraints are most diagnostic, which algorithms fit which patterns, how to balance information gathering vs time efficiency, and how to recover from wrong hypotheses across three progressively nuanced task objectives.

The core skill — **reasoning under partial information with costly actions** — is one of the most transferable capabilities an RL agent can develop. CP-Arena provides a structured, verifiable, fully automated environment to measure and improve it.

---

## Tech Stack

OpenEnv environment framework, FastAPI and Uvicorn for the HTTP server, Pydantic for typed models, Docker for containerized deployment, and Hugging Face Spaces for hosting.

---