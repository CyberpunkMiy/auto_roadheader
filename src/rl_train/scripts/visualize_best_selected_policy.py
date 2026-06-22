import argparse
import os
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


RL_TRAIN = r"D:\vscode_project\chuanlian _No\RL_juejin\src\rl_train"
sys.path.append(RL_TRAIN)

from rl_env import RoadheaderDiggingEnv


DEFAULT_MODEL = os.path.join(RL_TRAIN, "models", "roadheader_best_selected.zip")
DEFAULT_VEC = os.path.join(RL_TRAIN, "models", "vecnormalize_best_selected.pkl")


def make_env(max_steps, render_pause):
    return RoadheaderDiggingEnv(
        render_mode=None,
        verbose=False,
        max_steps=max_steps,
        control_mode="kinematic",
        render_pause=render_pause,
        camera_lookat=(0.0, 3.55, 2.0),
        camera_distance=7.2,
        camera_azimuth=145.0,
        camera_elevation=-24.0,
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize the selected best roadheader policy.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--vecnormalize", default=DEFAULT_VEC)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--render-pause", type=float, default=0.025)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--spin-speed", type=float, default=18.0, help="Visual cutter spin in rad/s.")
    args = parser.parse_args()

    print(f"Loading model: {args.model}", flush=True)
    print(f"Loading VecNormalize: {args.vecnormalize}", flush=True)

    raw_env = make_env(args.max_steps, args.render_pause)
    venv = DummyVecEnv([lambda: raw_env])
    env = VecNormalize.load(args.vecnormalize, venv)
    env.training = False
    env.norm_reward = False
    model = PPO.load(args.model)

    obs = env.reset()
    done = np.array([False])
    total_reward = 0.0
    last_info = {}
    cutter_joint_id = mujoco.mj_name2id(raw_env.model, mujoco.mjtObj.mjOBJ_JOINT, "jiegetou_joint")
    cutter_qpos_addr = raw_env.model.jnt_qposadr[cutter_joint_id] if cutter_joint_id != -1 else -1
    spin_per_step = args.spin_speed * args.render_pause

    print("Starting MuJoCo rollout. Close the MuJoCo window to stop watching.", flush=True)
    try:
        for step in range(1, args.max_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            total_reward += float(reward[0])
            last_info = infos[0] if infos else {}
            if cutter_qpos_addr >= 0:
                raw_env.data.qpos[cutter_qpos_addr] += spin_per_step
                mujoco.mj_forward(raw_env.model, raw_env.data)
            raw_env.render()

            if step == 1 or step % args.print_every == 0 or last_info.get("voxels_removed", 0) > 0:
                print(
                    f"step={step:4d} "
                    f"removed_now={last_info.get('voxels_removed', 0):3d} "
                    f"remaining={last_info.get('remaining_voxels', '-') } "
                    f"removed_fraction={last_info.get('removed_fraction', 0.0):.4f}",
                    flush=True,
                )

            if bool(done[0]):
                break
    finally:
        time.sleep(1.0)
        env.close()

    print("\nRollout finished.", flush=True)
    print(f"total_reward={total_reward:.3f}", flush=True)
    print(f"remaining_voxels={last_info.get('remaining_voxels')}", flush=True)
    print(f"removed_fraction={last_info.get('removed_fraction')}", flush=True)
    print("Press Enter to close this PowerShell window.", flush=True)
    try:
        input()
    except EOFError:
        pass


if __name__ == "__main__":
    main()
