import argparse
import os
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(r"D:\vscode_project\chuanlian _No\RL_juejin\src\rl_train")

from rl_env import RoadheaderDiggingEnv


def make_env(max_steps):
    return RoadheaderDiggingEnv(
        render_mode=None,
        verbose=False,
        max_steps=max_steps,
        control_mode="kinematic",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vecnormalize", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=3000)
    args = parser.parse_args()

    venv = DummyVecEnv([lambda: make_env(args.max_steps)])
    env = VecNormalize.load(args.vecnormalize, venv)
    env.training = False
    env.norm_reward = False
    model = PPO.load(args.model)

    results = []
    for episode in range(args.episodes):
        obs = env.reset()
        done = np.array([False])
        total_reward = 0.0
        steps = 0
        last_info = {}
        while not bool(done[0]):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            total_reward += float(reward[0])
            last_info = infos[0]
            steps += 1

        result = {
            "episode": episode + 1,
            "steps": steps,
            "reward": total_reward,
            "remaining_voxels": last_info.get("remaining_voxels"),
            "removed_fraction": last_info.get("removed_fraction"),
            "success": last_info.get("remaining_voxels") == 0,
        }
        results.append(result)
        print(result)

    remaining = [r["remaining_voxels"] for r in results if r["remaining_voxels"] is not None]
    removed = [r["removed_fraction"] for r in results if r["removed_fraction"] is not None]
    print(
        "summary:",
        {
            "episodes": len(results),
            "max_steps": args.max_steps,
            "successes": sum(1 for r in results if r["success"]),
            "mean_remaining": float(np.mean(remaining)) if remaining else None,
            "best_remaining": int(np.min(remaining)) if remaining else None,
            "mean_removed_fraction": float(np.mean(removed)) if removed else None,
            "best_removed_fraction": float(np.max(removed)) if removed else None,
        },
    )


if __name__ == "__main__":
    main()
