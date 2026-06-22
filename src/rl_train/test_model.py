import argparse
import os
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from rl_env import RoadheaderDiggingEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Run a trained roadheader PPO policy.")
    parser.add_argument("--model", type=str, default=os.path.join(current_dir, "models", "roadheader_final.zip"))
    parser.add_argument("--vecnormalize", type=str, default=os.path.join(current_dir, "models", "vecnormalize.pkl"))
    parser.add_argument("--sleep", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        return

    print(f"Loading model: {args.model}")
    model = PPO.load(args.model)

    venv = DummyVecEnv([lambda: RoadheaderDiggingEnv(render_mode="human", verbose=True)])
    if os.path.exists(args.vecnormalize):
        print(f"Loading normalization stats: {args.vecnormalize}")
        env = VecNormalize.load(args.vecnormalize, venv)
        env.training = False
        env.norm_reward = False
    else:
        print("Normalization stats not found; running without VecNormalize.")
        env = venv

    obs = env.reset()
    total_reward = 0.0
    episode = 1
    step = 0
    print("Starting rollout. Close the MuJoCo viewer to stop.")

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = env.step(action)
        total_reward += float(reward[0])
        step += 1

        info = infos[0] if infos else {}
        if info.get("voxels_removed", 0) > 0:
            print(
                f"episode={episode} step={step} "
                f"removed={info['voxels_removed']} remaining={info['remaining_voxels']}"
            )

        if args.sleep > 0:
            time.sleep(args.sleep)

        if bool(done[0]):
            print(f"Episode {episode} finished. Reward: {total_reward:.2f}")
            obs = env.reset()
            total_reward = 0.0
            episode += 1
            step = 0


if __name__ == "__main__":
    main()
