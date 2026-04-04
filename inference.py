import asyncio
import os
import textwrap
from typing import List
from openai import OpenAI
from cp_arena_env.client import CpArenaEnv
from cp_arena_env.models import CpArenaAction

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
IMAGE_NAME = os.getenv("IMAGE_NAME")
BENCHMARK = "cp_arena_env"
MAX_STEPS = 15

TASKS = [
    "algorithm_selection",
    "complexity_optimization",
    "problem_classification",
]

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI competitive programmer solving CP problems.
    Choose action IDs (integers 0-13).

    Actions:
    0: reveal_N
    1: reveal_time_limit
    2: reveal_memory
    3: reveal_tags
    4: reveal_example
    5: test_greedy
    6: test_dp
    7: test_graph
    8: test_binary_search
    9: test_math
    10: test_string
    11: submit_linear
    12: submit_nlogn
    13: submit_quadratic

    Strategy:
    - Reveal N and time_limit first
    - Test the most likely algorithm
    - Submit correct complexity
    - Reply with ONLY a single integer (0-13).
""").strip()


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_action(client, step, last_message, history):
    history_block = "\n".join(history[-5:]) if history else "None"
    user_prompt = f"""
Step: {step}
Last observation: {last_message}
History:
{history_block}

Choose your next action (reply with a single integer 0-13):
""".strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=10,
        )
        text = (completion.choices[0].message.content or "").strip()
        action_id = int(text.strip().split()[0])
        if 0 <= action_id <= 13:
            return action_id
        return 0
    except Exception as e:
        print(f"[DEBUG] Model error: {e}", flush=True)
        return 0


async def run_task(task_name: str):
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await CpArenaEnv.from_docker_image(IMAGE_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[str] = []

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task_name)
        last_message = result.observation.message

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_id = get_action(client, step, last_message, history)
            result = await env.step(CpArenaAction(action_id=action_id))

            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step
            last_message = obs.message

            log_step(step=step, action=action_id, reward=reward, done=done, error=None)
            history.append(f"Step {step}: action={action_id} reward={reward:+.2f} msg={obs.message[:80]}")

            if done:
                break

        total = sum(rewards)
        score = (total + 100) / 300
        score = max(0.0, min(1.0, score))
        success = score >= 0.1

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main():
    for task in TASKS:
        await run_task(task)


if __name__ == "__main__":
    asyncio.run(main())