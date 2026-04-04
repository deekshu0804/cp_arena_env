---
title: CP Arena Env
emoji: "\U0001F3C6"
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
base_path: /web
---

# CP-Arena: Competitive Programming RL Environment

A **partially observable reinforcement learning environment** where an AI agent learns to solve competitive programming problems the way a real human does — by gathering clues, forming hypotheses, and submitting under time pressure.

---

## What This Environment Does

Most RL environments give agents complete information upfront. CP-Arena does not.

The agent starts blind. Every episode, a hidden CP problem is loaded. The agent must spend actions to reveal information, test reasoning hypotheses about which algorithm fits, commit to a solution, and get a verdict — all under time pressure.

This models the real decision process of a competitive programmer during a contest.

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
- Efficiency score = (5 - reveals_used) * 10
- Wrong classification: -25
- Encourages sharp inference over brute-force information gathering

---

## Normalized Score

All tasks return a normalized score in [0.0, 1.0]:

```
score = (total_reward + 100) / 300
score = clamp(score, 0.0, 1.0)
```

---

## Environment Flow

```
RESET(task="algorithm_selection")
  agent sees: difficulty and task description only

Agent reveals N scale          costs time (-2)
Agent reveals time pressure    costs time (-4)
Agent tests graph hypothesis   signal returned (+3 or -3)
Agent submits O(N log N)       verdict: AC

DONE  normalized score returned
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

### Key Design Decisions

Exponential info cost forces the agent to learn which constraints are most diagnostic. Revealing everything is heavily penalized.

Reasoning signals make hypothesis testing meaningful. Each test action returns a plausibility signal (+3 or -3) rather than being a label.

Consistency penalty penalizes illogical decisions like choosing quadratic complexity after testing a graph algorithm.

Efficiency bonus rewards agents that solve correctly with fewer information reveals.

---

## Dataset

50 competitive programming problems spanning difficulties 800 to 2000 on the Codeforces rating scale, covering algorithm types greedy, dp, graph, binary_search, math, string, and brute_force, with complexities O(N), O(N log N), O(N^2), across small, medium, and large input sizes.

Sample problem format:
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

## Episode Example Trace

```
[RESET] Task: algorithm_selection | Problem: cf_031 | Difficulty: 1600

Step 1 | Action: reveal_N         | Reward: -2.0  | N scale: large
Step 2 | Action: reveal_time_limit | Reward: -4.0  | Time: tight
Step 3 | Action: test_graph        | Reward: +3.0  | PLAUSIBLE
Step 4 | Action: submit_nlogn      | Reward: +95.0 | ACCEPTED

Total reward: 92.0 | Score: 0.64 | Steps: 4
```

---

## Environment Design Philosophy

CP-Arena models the decision process of a competitive programmer, not the act of writing code. The agent learns which constraints are most diagnostic, which algorithms fit which patterns, how to balance information gathering vs time efficiency, and how to recover from wrong hypotheses across three progressively nuanced task objectives.

---

## Tech Stack

OpenEnv environment framework, FastAPI and Uvicorn for the HTTP server, Pydantic for typed models, Docker for containerized deployment, and Hugging Face Spaces for hosting.

---

## License

BSD-style license following OpenEnv conventions.
