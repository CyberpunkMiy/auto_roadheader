import argparse
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


PROJECT = Path(r"D:\vscode_project\chuanlian _No\RL_juejin")
RL_TRAIN = PROJECT / "src" / "rl_train"
MODELS = RL_TRAIN / "models"
sys.path.append(str(RL_TRAIN))

from rl_env import RoadheaderDiggingEnv


def make_env(max_steps):
    return RoadheaderDiggingEnv(
        render_mode=None,
        verbose=False,
        max_steps=max_steps,
        control_mode="kinematic",
    )


def evaluate(model_path, vec_path, episodes, max_steps):
    venv = DummyVecEnv([lambda: make_env(max_steps)])
    env = VecNormalize.load(str(vec_path), venv)
    env.training = False
    env.norm_reward = False
    model = PPO.load(str(model_path))

    results = []
    for _ in range(episodes):
        obs = env.reset()
        done = np.array([False])
        total_reward = 0.0
        steps = 0
        info = {}
        while not bool(done[0]):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            total_reward += float(reward[0])
            info = infos[0]
            steps += 1
        results.append(
            {
                "steps": steps,
                "reward": total_reward,
                "remaining": int(info.get("remaining_voxels", -1)),
                "removed_fraction": float(info.get("removed_fraction", 0.0)),
                "success": info.get("remaining_voxels") == 0,
            }
        )
    env.close()
    return results


def summarize(name, model, vec, results):
    remaining = [r["remaining"] for r in results]
    removed = [r["removed_fraction"] for r in results]
    rewards = [r["reward"] for r in results]
    return {
        "name": name,
        "model": str(model),
        "vecnormalize": str(vec),
        "episodes": len(results),
        "successes": sum(r["success"] for r in results),
        "best_remaining": min(remaining),
        "mean_remaining": float(np.mean(remaining)),
        "best_removed_fraction": float(np.max(removed)),
        "mean_removed_fraction": float(np.mean(removed)),
        "mean_reward": float(np.mean(rewards)),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2000)
    args = parser.parse_args()

    candidates = [
        ("best_model_with_1800k_norm", MODELS / "best_model.zip", MODELS / "roadheader_ppo_vecnormalize_1800000_steps.pkl"),
        ("1800k", MODELS / "roadheader_ppo_1800000_steps.zip", MODELS / "roadheader_ppo_vecnormalize_1800000_steps.pkl"),
        ("2000k", MODELS / "roadheader_ppo_2000000_steps.zip", MODELS / "roadheader_ppo_vecnormalize_2000000_steps.pkl"),
        ("2200k", MODELS / "roadheader_ppo_2200000_steps.zip", MODELS / "roadheader_ppo_vecnormalize_2200000_steps.pkl"),
        ("2400k", MODELS / "roadheader_ppo_2400000_steps.zip", MODELS / "roadheader_ppo_vecnormalize_2400000_steps.pkl"),
        ("2600k", MODELS / "roadheader_ppo_2600000_steps.zip", MODELS / "roadheader_ppo_vecnormalize_2600000_steps.pkl"),
        ("2800k", MODELS / "roadheader_ppo_2800000_steps.zip", MODELS / "roadheader_ppo_vecnormalize_2800000_steps.pkl"),
        ("3000k", MODELS / "roadheader_ppo_3000000_steps.zip", MODELS / "roadheader_ppo_vecnormalize_3000000_steps.pkl"),
        ("final", MODELS / "roadheader_final.zip", MODELS / "vecnormalize.pkl"),
    ]

    summaries = []
    for name, model, vec in candidates:
        if not model.exists() or not vec.exists():
            print(f"SKIP {name}: missing file")
            continue
        print(f"Evaluating {name}...", flush=True)
        results = evaluate(model, vec, args.episodes, args.max_steps)
        summary = summarize(name, model, vec, results)
        summaries.append(summary)
        print(summary, flush=True)

    summaries.sort(key=lambda s: (s["best_remaining"], s["mean_remaining"], -s["mean_reward"]))
    print("\nRANKING")
    print("=" * 80)
    for i, s in enumerate(summaries, 1):
        print(
            f"{i}. {s['name']}: best_remaining={s['best_remaining']} "
            f"mean_remaining={s['mean_remaining']:.1f} "
            f"best_removed={s['best_removed_fraction']:.5f} "
            f"successes={s['successes']}/{s['episodes']} "
            f"mean_reward={s['mean_reward']:.2f}"
        )
    if summaries:
        best = summaries[0]
        print("\nBEST")
        print(best)


if __name__ == "__main__":
    main()
