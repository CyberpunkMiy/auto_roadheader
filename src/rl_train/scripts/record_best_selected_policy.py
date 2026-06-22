import argparse
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import imageio.v2 as imageio
import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


RL_TRAIN = r"D:\vscode_project\chuanlian _No\RL_juejin\src\rl_train"
sys.path.append(RL_TRAIN)

from rl_env import RoadheaderDiggingEnv


DEFAULT_MODEL = os.path.join(RL_TRAIN, "models", "roadheader_best_selected.zip")
DEFAULT_VEC = os.path.join(RL_TRAIN, "models", "vecnormalize_best_selected.pkl")
DEFAULT_OUT = (
    Path(RL_TRAIN)
    / "logs"
    / "best_selected_visual_eval_half_speed.mp4"
)


def make_env(max_steps):
    return RoadheaderDiggingEnv(
        render_mode=None,
        verbose=False,
        max_steps=max_steps,
        control_mode="kinematic",
    )


def main():
    parser = argparse.ArgumentParser(description="Record the selected best roadheader policy.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--vecnormalize", default=DEFAULT_VEC)
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--spin-speed", type=float, default=28.0)
    parser.add_argument("--print-every", type=int, default=50)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    raw_env = make_env(args.max_steps)
    venv = DummyVecEnv([lambda: raw_env])
    env = VecNormalize.load(args.vecnormalize, venv)
    env.training = False
    env.norm_reward = False
    model = PPO.load(args.model)

    renderer = mujoco.Renderer(raw_env.model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = np.array([0.0, 3.55, 2.0], dtype=np.float64)
    camera.distance = 7.2
    camera.azimuth = 145.0
    camera.elevation = -24.0

    cutter_joint_id = mujoco.mj_name2id(raw_env.model, mujoco.mjtObj.mjOBJ_JOINT, "jiegetou_joint")
    cutter_qpos_addr = raw_env.model.jnt_qposadr[cutter_joint_id] if cutter_joint_id != -1 else -1
    spin_per_frame = args.spin_speed / max(args.fps, 1)

    obs = env.reset()
    done = np.array([False])
    total_reward = 0.0
    last_info = {}

    print(f"Recording to: {output}", flush=True)
    print(f"Playback fps: {args.fps} (half-speed visual playback)", flush=True)

    with imageio.get_writer(str(output), fps=args.fps, codec="libx264", quality=8) as writer:
        for step in range(1, args.max_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            total_reward += float(reward[0])
            last_info = infos[0] if infos else {}

            if cutter_qpos_addr >= 0:
                raw_env.data.qpos[cutter_qpos_addr] += spin_per_frame
                mujoco.mj_forward(raw_env.model, raw_env.data)

            renderer.update_scene(raw_env.data, camera=camera)
            frame = renderer.render()
            writer.append_data(frame)

            if step == 1 or step % args.print_every == 0 or bool(done[0]):
                print(
                    f"step={step:4d} "
                    f"remaining={last_info.get('remaining_voxels', '-')} "
                    f"removed_fraction={last_info.get('removed_fraction', 0.0):.4f}",
                    flush=True,
                )

            if bool(done[0]):
                break

    renderer.close()
    env.close()

    print("Recording finished.", flush=True)
    print(f"output={output}", flush=True)
    print(f"total_reward={total_reward:.3f}", flush=True)
    print(f"remaining_voxels={last_info.get('remaining_voxels')}", flush=True)
    print(f"removed_fraction={last_info.get('removed_fraction')}", flush=True)


if __name__ == "__main__":
    main()
