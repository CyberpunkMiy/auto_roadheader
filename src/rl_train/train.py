import argparse
import importlib.util
import os
import sys
import time
from typing import Callable

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from rl_env import RoadheaderDiggingEnv

MONITOR_INFO_KEYWORDS = (
    "remaining_voxels",
    "removed_fraction",
    "target_distance",
    "actual_tip_step",
    "is_stuck",
    "ik_failed",
)


class ExcavationMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._last_logged_episodes = 0

    def _on_step(self):
        ep_info_buffer = getattr(self.model, "ep_info_buffer", None)
        if not ep_info_buffer:
            return True

        total_episodes = len(getattr(self.model, "ep_info_buffer", []))
        if hasattr(self.model, "_episode_num"):
            total_episodes = int(self.model._episode_num)
        if total_episodes == self._last_logged_episodes:
            return True
        self._last_logged_episodes = total_episodes

        episodes = list(ep_info_buffer)

        def values(key):
            return [float(ep[key]) for ep in episodes if key in ep]

        removed = values("removed_fraction")
        remaining = values("remaining_voxels")
        step_len = values("actual_tip_step")
        target_distance = values("target_distance")
        stuck = values("is_stuck")
        ik_failed = values("ik_failed")

        if removed:
            self.logger.record("excavation/removed_fraction_mean", sum(removed) / len(removed))
            self.logger.record("excavation/removed_fraction_best", max(removed))
            self.logger.record("excavation/success_rate", sum(v >= 0.999 for v in removed) / len(removed))
        if remaining:
            self.logger.record("excavation/remaining_voxels_mean", sum(remaining) / len(remaining))
            self.logger.record("excavation/remaining_voxels_best", min(remaining))
        if step_len:
            self.logger.record("excavation/actual_tip_step_mean", sum(step_len) / len(step_len))
        if target_distance:
            self.logger.record("excavation/target_distance_mean", sum(target_distance) / len(target_distance))
        if stuck:
            self.logger.record("excavation/stuck_rate", sum(stuck) / len(stuck))
        if ik_failed:
            self.logger.record("excavation/ik_failed_rate", sum(ik_failed) / len(ik_failed))

        return True


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def make_env(rank, seed=0, args=None, log_dir=None):
    def _init():
        env = RoadheaderDiggingEnv(
            render_mode=None,
            fixed_tip_step=args.fixed_tip_step,
            frame_skip=args.frame_skip,
            control_mode=args.control_mode,
            max_steps=args.max_steps,
            target_k=args.target_k,
            cut_reward_scale=args.cut_reward_scale,
            time_penalty=args.time_penalty,
            empty_cut_penalty=args.empty_cut_penalty,
            stuck_penalty=args.stuck_penalty,
            progress_reward_scale=args.progress_reward_scale,
            completion_bonus=args.completion_bonus,
            early_finish_bonus_scale=args.early_finish_bonus_scale,
            verbose=False,
        )
        if log_dir is not None:
            env = Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                info_keywords=MONITOR_INFO_KEYWORDS,
            )
        else:
            env = Monitor(env, info_keywords=MONITOR_INFO_KEYWORDS)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def build_vec_env(args, log_dir):
    env_fns = [make_env(i, args.seed, args, log_dir) for i in range(args.n_envs)]
    if args.vec_env == "subproc" and args.n_envs > 1:
        return SubprocVecEnv(env_fns, start_method="spawn")
    return DummyVecEnv(env_fns)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO for roadheader voxel excavation.")
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="subproc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-tip-step", type=float, default=0.05)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--control-mode", choices=["kinematic", "actuator"], default="kinematic")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--target-k", type=int, default=80)
    parser.add_argument("--cut-reward-scale", type=float, default=1.0)
    parser.add_argument("--time-penalty", type=float, default=0.02)
    parser.add_argument("--empty-cut-penalty", type=float, default=0.05)
    parser.add_argument("--stuck-penalty", type=float, default=0.25)
    parser.add_argument("--progress-reward-scale", type=float, default=0.3)
    parser.add_argument("--completion-bonus", type=float, default=100.0)
    parser.add_argument("--early-finish-bonus-scale", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--save-freq", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--resume-vecnormalize", type=str, default="")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    models_dir = os.path.join(current_dir, "models")
    logs_dir = os.path.join(current_dir, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(f"Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    print(f"Vector env: {args.vec_env} x {args.n_envs}")
    print(f"Control mode: {args.control_mode}")
    print(f"Fixed cutter step: {args.fixed_tip_step} m per decision")
    print("=" * 60)

    tensorboard_log = None
    if args.tensorboard:
        has_tensorboard = importlib.util.find_spec("tensorboard") is not None
        if has_tensorboard:
            tensorboard_log = logs_dir
        else:
            print("TensorBoard is not installed; continuing without tensorboard logging.")

    env = build_vec_env(args, logs_dir)
    resume_vecnormalize = args.resume_vecnormalize
    if args.resume and not resume_vecnormalize:
        base, _ = os.path.splitext(args.resume)
        candidate = base.replace("roadheader_ppo_", "roadheader_ppo_vecnormalize_") + ".pkl"
        if os.path.exists(candidate):
            resume_vecnormalize = candidate

    if resume_vecnormalize:
        print(f"Loading VecNormalize from: {resume_vecnormalize}")
        env = VecNormalize.load(resume_vecnormalize, env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.995)

    eval_env = DummyVecEnv([make_env(10_000, args.seed, args, None)])
    if resume_vecnormalize:
        eval_env = VecNormalize.load(resume_vecnormalize, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=env, device=device)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            device=device,
            verbose=1,
            learning_rate=linear_schedule(args.learning_rate),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.005,
            clip_range=0.2,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // max(args.n_envs, 1), 1),
        save_path=models_dir,
        name_prefix="roadheader_ppo",
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=logs_dir,
        eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )
    excavation_metrics_callback = ExcavationMetricsCallback()

    print(f"Training for {args.timesteps:,} timesteps...")
    start_time = time.time()
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, eval_callback, excavation_metrics_callback],
            progress_bar=args.progress_bar,
            reset_num_timesteps=not bool(args.resume),
        )
    except KeyboardInterrupt:
        print("\nInterrupted; saving current model.")

    final_path = os.path.join(models_dir, "roadheader_final")
    norm_path = os.path.join(models_dir, "vecnormalize.pkl")
    model.save(final_path)
    env.save(norm_path)
    env.close()
    eval_env.close()

    print(f"Saved model: {final_path}.zip")
    print(f"Saved normalization stats: {norm_path}")
    print(f"Elapsed: {(time.time() - start_time) / 60:.2f} min")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()


