import os
import sys
import time

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    from control.trajectory_control_interactive import RoadheaderController
    from core.digging_system_mesh import MeshDiggingSystem
except ImportError as exc:
    raise ImportError(f"Failed to import project modules from {src_dir}: {exc}") from exc


class RoadheaderDiggingEnv(gym.Env):
    """
    MuJoCo/Gymnasium environment for roadheader voxel excavation.

    The policy controls only the direction of the cutter tip. Each environment
    step commands the same Cartesian displacement length, so the learning
    problem is: which direction should the fixed-speed cutter move next to
    clear all active voxels in the fewest steps?
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        xml_path=None,
        mesh_name="jiegetou_link",
        body_name="jiegetou_link",
        scene_body_name="voxel_target",
        render_mode=None,
        fixed_tip_step=0.05,
        frame_skip=5,
        control_mode="kinematic",
        max_steps=2000,
        target_k=80,
        cut_reward_scale=1.0,
        time_penalty=0.02,
        empty_cut_penalty=0.05,
        stuck_penalty=0.25,
        progress_reward_scale=0.3,
        completion_bonus=100.0,
        early_finish_bonus_scale=0.05,
        camera_lookat=(0.0, 2.8, 1.8),
        camera_distance=8.0,
        camera_azimuth=135.0,
        camera_elevation=-22.0,
        render_pause=0.0,
        verbose=False,
    ):
        super().__init__()

        self.xml_path = xml_path or os.path.join(src_dir, "output", "merged_result.xml")
        self.mesh_name = mesh_name
        self.body_name = body_name
        self.scene_body_name = scene_body_name
        self.render_mode = render_mode
        self.fixed_tip_step = float(fixed_tip_step)
        self.frame_skip = int(frame_skip)
        self.control_mode = control_mode
        self.max_steps = int(max_steps)
        self.target_k = int(target_k)
        self.cut_reward_scale = float(cut_reward_scale)
        self.time_penalty = float(time_penalty)
        self.empty_cut_penalty = float(empty_cut_penalty)
        self.stuck_penalty = float(stuck_penalty)
        self.progress_reward_scale = float(progress_reward_scale)
        self.completion_bonus = float(completion_bonus)
        self.early_finish_bonus_scale = float(early_finish_bonus_scale)
        self.camera_lookat = np.asarray(camera_lookat, dtype=np.float64)
        self.camera_distance = float(camera_distance)
        self.camera_azimuth = float(camera_azimuth)
        self.camera_elevation = float(camera_elevation)
        self.render_pause = float(render_pause)
        self.verbose = verbose

        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"MuJoCo XML not found: {self.xml_path}")

        if self.verbose:
            print(f"[RL_Env] Loading model: {self.xml_path}")

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.controller = RoadheaderController(self.model, self.data)
        self.digger = MeshDiggingSystem(
            self.model,
            self.data,
            mesh_name=self.mesh_name,
            scene_body_name=self.scene_body_name,
            verbose=self.verbose,
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation layout:
        # tip position (3), normalized joint positions (3), target vector (3),
        # target distance (1), unit direction to target (3), remaining fraction (1),
        # previous commanded direction (3), normalized step index (1) = 18 dims.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.initial_voxel_count = 1
        self.last_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.last_target_distance = 0.0
        self.viewer = None

        self._joint_qpos_addr = np.array(
            [self.model.jnt_qposadr[jid] for jid in self.controller.joint_ids],
            dtype=np.int32,
        )
        self._joint_ranges = np.array(
            [self.model.jnt_range[jid] for jid in self.controller.joint_ids],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.digger.reset()
        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0
        self.initial_voxel_count = max(self.digger.remaining_count, 1)
        self.last_direction = self._direction_to_target()
        self.last_target_distance = self._target_distance()

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        pos_before = self.controller.get_current_site_pos()
        target_before = self._get_target_center(pos_before)
        dist_before = float(np.linalg.norm(target_before - pos_before))

        direction = self._normalize_action(action)
        commanded_target = pos_before + direction * self.fixed_tip_step

        q_cmd = self.controller.solve_ik(commanded_target)
        ik_failed = q_cmd is None
        if not ik_failed and self.control_mode == "kinematic":
            self._apply_kinematic_command(q_cmd)
        elif not ik_failed:
            self.controller.control_actuators(q_cmd)
            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)

        pos_after = self.controller.get_current_site_pos()
        actual_dist = float(np.linalg.norm(pos_after - pos_before))
        is_stuck = actual_dist < self.fixed_tip_step * 0.1

        voxels_removed = self.digger.perform_cutting(self.body_name)
        remaining = self.digger.remaining_count
        removed_fraction = 1.0 - (remaining / max(self.initial_voxel_count, 1))

        target_after = self._get_target_center(pos_after)
        dist_after = float(np.linalg.norm(target_after - pos_after))
        distance_delta = dist_before - dist_after

        reward = 0.0
        reward += self.cut_reward_scale * voxels_removed
        reward += self.progress_reward_scale * (distance_delta / max(self.fixed_tip_step, 1e-6))
        reward -= self.time_penalty

        if voxels_removed == 0:
            reward -= self.empty_cut_penalty
        if is_stuck or ik_failed:
            reward -= self.stuck_penalty

        terminated = remaining == 0
        truncated = self.current_step >= self.max_steps

        if terminated:
            steps_left = max(self.max_steps - self.current_step, 0)
            reward += self.completion_bonus + steps_left * self.early_finish_bonus_scale
            if self.verbose:
                print(f"Episode finished in {self.current_step} steps.")

        self.last_direction = direction.astype(np.float32)
        self.last_target_distance = dist_after

        if self.render_mode == "human":
            self.render()

        info = {
            "voxels_removed": int(voxels_removed),
            "remaining_voxels": int(remaining),
            "removed_fraction": float(removed_fraction),
            "target_distance": dist_after,
            "actual_tip_step": actual_dist,
            "fixed_tip_step": self.fixed_tip_step,
            "is_stuck": bool(is_stuck),
            "ik_failed": bool(ik_failed),
            "control_mode": self.control_mode,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _apply_kinematic_command(self, q_cmd):
        for value, jid in zip(q_cmd, self.controller.joint_ids):
            qpos_addr = self.model.jnt_qposadr[jid]
            self.data.qpos[qpos_addr] = value
        mujoco.mj_forward(self.model, self.data)

    def _normalize_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        norm = float(np.linalg.norm(action))
        if norm > 1e-6:
            return action / norm

        fallback = self._direction_to_target()
        if np.linalg.norm(fallback) > 1e-6:
            return fallback
        return self.last_direction.copy()

    def _direction_to_target(self):
        tip_pos = self.controller.get_current_site_pos()
        target = self._get_target_center(tip_pos)
        vec = target - tip_pos
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return self.last_direction.copy()
        return (vec / norm).astype(np.float32)

    def _get_target_center(self, head_pos=None):
        if head_pos is None:
            head_pos = self.controller.get_current_site_pos()
        return self.digger.get_local_target(head_pos, k=self.target_k)

    def _target_distance(self):
        tip_pos = self.controller.get_current_site_pos()
        return float(np.linalg.norm(self._get_target_center(tip_pos) - tip_pos))

    def _get_normalized_joint_positions(self):
        q = self.data.qpos[self._joint_qpos_addr].astype(np.float32)
        low = self._joint_ranges[:, 0]
        high = self._joint_ranges[:, 1]
        center = (low + high) * 0.5
        half_range = np.maximum((high - low) * 0.5, 1e-6)
        return np.clip((q - center) / half_range, -1.5, 1.5)

    def _get_obs(self):
        tip_pos = self.controller.get_current_site_pos().astype(np.float32)
        target_center = self._get_target_center(tip_pos).astype(np.float32)
        target_vec = target_center - tip_pos
        target_dist = float(np.linalg.norm(target_vec))
        target_dir = target_vec / target_dist if target_dist > 1e-6 else np.zeros(3, dtype=np.float32)
        remaining_fraction = self.digger.remaining_count / max(self.initial_voxel_count, 1)
        step_fraction = self.current_step / max(self.max_steps, 1)

        obs = np.concatenate(
            [
                tip_pos,
                self._get_normalized_joint_positions(),
                target_vec.astype(np.float32),
                np.array([target_dist], dtype=np.float32),
                target_dir.astype(np.float32),
                np.array([remaining_fraction], dtype=np.float32),
                self.last_direction.astype(np.float32),
                np.array([step_fraction], dtype=np.float32),
            ]
        )
        return obs.astype(np.float32)

    def render(self):
        if self.viewer is None:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.lookat[:] = self.camera_lookat
            self.viewer.cam.distance = self.camera_distance
            self.viewer.cam.azimuth = self.camera_azimuth
            self.viewer.cam.elevation = self.camera_elevation
        self.viewer.sync()
        if self.render_pause > 0:
            time.sleep(self.render_pause)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
