import mujoco
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


class MeshDiggingSystem:
    def __init__(
        self,
        model,
        data,
        mesh_name,
        scene_body_name="imported_scene",
        voxel_xml_prefix="vx_",
        spacing=0.14,
        x_start=2.0,
        clean_threshold=(1.0, 0.5, 0.5),
        verbose=True,
    ):
        self.model = model
        self.data = data
        self.mesh_name = mesh_name
        self.scene_body_name = scene_body_name
        self.voxel_xml_prefix = voxel_xml_prefix
        self.spacing = float(spacing)
        self.box_size = self.spacing / 2.0
        self.wall_x_start = float(x_start)
        self.verbose = verbose
        self.step_counter = 0
        self.last_head_pos = None
        self.last_head_mat = None
        self._cutting_body_cache = {}

        self._init_scene_transform()
        self._init_cutter_mesh(clean_threshold)
        self._index_voxels()

    @property
    def remaining_count(self):
        return int(np.count_nonzero(self.voxel_mask))

    def _log(self, message):
        if self.verbose:
            print(message)

    def _init_scene_transform(self):
        self.scene_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.scene_body_name,
        )
        if self.scene_body_id != -1:
            mujoco.mj_forward(self.model, self.data)
            self.scene_pos = self.data.body(self.scene_body_id).xpos.copy()
            self.scene_rot = self.data.body(self.scene_body_id).xmat.reshape(3, 3).copy()
            self.scene_rot_inv = self.scene_rot.T
            self._log(f"[Digging] Scene anchor: {self.scene_body_name}")
        else:
            self.scene_pos = np.zeros(3, dtype=np.float64)
            self.scene_rot = np.eye(3, dtype=np.float64)
            self.scene_rot_inv = np.eye(3, dtype=np.float64)
            self._log(f"[Digging] Scene anchor not found: {self.scene_body_name}; using world frame.")

    def _init_cutter_mesh(self, clean_threshold):
        mesh_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MESH, self.mesh_name)
        if mesh_id == -1:
            raise ValueError(f"Mesh not found: {self.mesh_name}")

        vert_adr = self.model.mesh_vertadr[mesh_id]
        vert_num = self.model.mesh_vertnum[mesh_id]
        raw_verts = self.model.mesh_vert[vert_adr : vert_adr + vert_num].reshape(-1, 3)

        radii = np.maximum(np.asarray(clean_threshold, dtype=np.float64), 1e-6)
        target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.mesh_name)
        joint_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        rotation_matrix = np.eye(3, dtype=np.float64)

        if target_body_id != -1 and self.model.body_jntnum[target_body_id] > 0:
            joint_adr = self.model.body_jntadr[target_body_id]
            joint_axis = self.model.jnt_axis[joint_adr].copy()

        target_axis = joint_axis / max(np.linalg.norm(joint_axis), 1e-6)
        source_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        if not np.allclose(target_axis, source_axis):
            rot_axis = np.cross(source_axis, target_axis)
            sin_theta = np.linalg.norm(rot_axis)
            cos_theta = np.dot(source_axis, target_axis)
            if sin_theta < 1e-6:
                if cos_theta < 0:
                    rotation_matrix = Rotation.from_euler("y", 180, degrees=True).as_matrix()
            else:
                rot_axis /= sin_theta
                theta = np.arctan2(sin_theta, cos_theta)
                rotation_matrix = Rotation.from_rotvec(rot_axis * theta).as_matrix()

        verts_to_check = Rotation.from_matrix(rotation_matrix).inv().apply(raw_verts)
        normalized = verts_to_check / radii
        valid_mask = np.sum(normalized * normalized, axis=1) <= 1.0
        self.mesh_verts = raw_verts[valid_mask] if len(raw_verts) else raw_verts

        if len(self.mesh_verts) == 0:
            self.max_radius = 0.1
            self.kdtree = None
            self._log("[Digging] Cutter mesh has no valid vertices after cleaning.")
            return

        self.max_radius = float(np.max(np.linalg.norm(self.mesh_verts, axis=1)))
        self.kdtree = cKDTree(self.mesh_verts)
        self._log(f"[Digging] Cutter mesh ready: {len(self.mesh_verts)} valid vertices.")

    def _index_voxels(self):
        mujoco.mj_forward(self.model, self.data)

        self.voxel_index = {}
        self.voxel_key_to_idx = {}
        self.idx_to_key = []
        self.idx_to_body_id = []
        centers = []
        geom_ids = []

        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not name or not name.startswith(self.voxel_xml_prefix):
                continue

            parts = name.split("_")
            if len(parts) < 4:
                continue

            try:
                key = (int(parts[-3]), int(parts[-2]), int(parts[-1]))
            except ValueError:
                continue

            idx = len(centers)
            self.voxel_index[key] = body_id
            self.voxel_key_to_idx[key] = idx
            self.idx_to_key.append(key)
            self.idx_to_body_id.append(body_id)
            centers.append(self.data.body(body_id).xpos.copy())

            geom_id = self.model.body_geomadr[body_id]
            geom_ids.append(geom_id if geom_id != -1 else -1)

        self.active_voxels = set(self.voxel_index.keys())
        if centers:
            self.voxel_centers_cache = np.asarray(centers, dtype=np.float32)
        else:
            self.voxel_centers_cache = np.empty((0, 3), dtype=np.float32)
        self.voxel_mask = np.ones(len(self.voxel_centers_cache), dtype=bool)

        self.geom_ids = np.asarray(geom_ids, dtype=np.int32)
        valid_geom_ids = self.geom_ids[self.geom_ids >= 0]
        self._valid_geom_mask = self.geom_ids >= 0
        self._valid_geom_ids = valid_geom_ids

        self._initial_geom_size = self.model.geom_size[valid_geom_ids].copy()
        self._initial_geom_rgba = self.model.geom_rgba[valid_geom_ids].copy()
        self._initial_geom_conaffinity = self.model.geom_conaffinity[valid_geom_ids].copy()
        self._initial_geom_contype = self.model.geom_contype[valid_geom_ids].copy()

        self._log(f"[Digging] Indexed {len(self.active_voxels)} voxels.")

    def world_to_local_grid(self, x, y, z):
        p_world = np.array([x, y, z], dtype=np.float64)
        p_local = self.scene_rot_inv @ (p_world - self.scene_pos)
        k = int(round((p_local[0] - self.wall_x_start - self.box_size) / self.spacing))
        i = int(round((p_local[2] - self.box_size) / self.spacing))
        j = int(round(p_local[1] / self.spacing))
        return k, i, j

    def _candidate_voxels(self, head_pos):
        center_k, center_i, center_j = self.world_to_local_grid(*head_pos)
        search_range = int(np.ceil(self.max_radius / self.spacing)) + 1

        candidate_keys = []
        candidate_indices = []
        candidate_positions = []

        for dk in range(-search_range, search_range + 1):
            for di in range(-search_range, search_range + 1):
                for dj in range(-search_range, search_range + 1):
                    key = (center_k + dk, center_i + di, center_j + dj)
                    if key not in self.active_voxels:
                        continue
                    idx = self.voxel_key_to_idx[key]
                    candidate_keys.append(key)
                    candidate_indices.append(idx)
                    candidate_positions.append(self.voxel_centers_cache[idx])

        if not candidate_positions:
            return [], np.empty(0, dtype=np.int32), np.empty((0, 3), dtype=np.float32)

        return (
            candidate_keys,
            np.asarray(candidate_indices, dtype=np.int32),
            np.asarray(candidate_positions, dtype=np.float32),
        )

    def _execute_single_cut(self, head_pos, head_mat, tolerance):
        if self.kdtree is None or self.remaining_count == 0:
            return 0

        candidate_keys, candidate_indices, candidate_positions = self._candidate_voxels(head_pos)
        if len(candidate_keys) == 0:
            return 0

        voxels_in_head_frame = (candidate_positions - head_pos) @ head_mat
        distances, _ = self.kdtree.query(voxels_in_head_frame, k=1)
        hit_candidate_indices = np.flatnonzero(distances <= tolerance)

        removed = 0
        for hit_i in hit_candidate_indices:
            key = candidate_keys[hit_i]
            cache_idx = candidate_indices[hit_i]
            if key not in self.active_voxels:
                continue

            geom_id = self.geom_ids[cache_idx]
            if geom_id >= 0:
                self.model.geom_size[geom_id] = 0.0
                self.model.geom_rgba[geom_id] = 0.0
                self.model.geom_conaffinity[geom_id] = 0
                self.model.geom_contype[geom_id] = 0

            self.active_voxels.remove(key)
            self.voxel_mask[cache_idx] = False
            removed += 1

        return removed

    def _resolve_cutting_frame(self, cutting_body_name):
        cached = self._cutting_body_cache.get(cutting_body_name)
        if cached is None:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cutting_body_name)
            if body_id != -1:
                cached = ("body", body_id)
            else:
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, cutting_body_name)
                if site_id == -1:
                    return None, None
                cached = ("site", site_id)
            self._cutting_body_cache[cutting_body_name] = cached

        frame_type, frame_id = cached
        if frame_type == "body":
            return (
                self.data.body(frame_id).xpos.copy(),
                self.data.body(frame_id).xmat.reshape(3, 3).copy(),
            )
        return (
            self.data.site_xpos[frame_id].copy(),
            self.data.site_xmat[frame_id].reshape(3, 3).copy(),
        )

    def perform_cutting(self, cutting_body_name, tolerance=0.05):
        self.step_counter += 1
        current_pos, current_mat = self._resolve_cutting_frame(cutting_body_name)
        if current_pos is None:
            return 0

        if self.last_head_pos is None:
            self.last_head_pos = current_pos
            self.last_head_mat = current_mat
            return self._execute_single_cut(current_pos, current_mat, tolerance)

        dist = float(np.linalg.norm(current_pos - self.last_head_pos))
        interpolation_step = self.spacing * 0.5

        if dist < interpolation_step:
            removed = self._execute_single_cut(current_pos, current_mat, tolerance)
        else:
            removed = 0
            num_steps = min(int(np.ceil(dist / interpolation_step)), 10)
            for i in range(1, num_steps + 1):
                alpha = i / num_steps
                interp_pos = self.last_head_pos + (current_pos - self.last_head_pos) * alpha
                interp_mat = self.last_head_mat + (current_mat - self.last_head_mat) * alpha
                removed += self._execute_single_cut(interp_pos, interp_mat, tolerance)

        self.last_head_pos = current_pos
        self.last_head_mat = current_mat
        return removed

    def get_local_target(self, head_pos, k=80):
        if self.remaining_count == 0:
            return np.zeros(3, dtype=np.float32)

        active_points = self.voxel_centers_cache[self.voxel_mask]
        if len(active_points) == 0:
            return np.zeros(3, dtype=np.float32)
        if len(active_points) <= k:
            return np.mean(active_points, axis=0)

        dist_sq = np.sum((active_points - head_pos) ** 2, axis=1)
        nearest_indices = np.argpartition(dist_sq, k)[:k]
        return np.mean(active_points[nearest_indices], axis=0)

    def get_remaining_voxel_center(self):
        if self.remaining_count == 0:
            return np.zeros(3, dtype=np.float32)
        return np.mean(self.voxel_centers_cache[self.voxel_mask], axis=0)

    def reset(self):
        self.active_voxels = set(self.voxel_index.keys())
        self.voxel_mask[:] = True

        if len(self._valid_geom_ids) > 0:
            self.model.geom_size[self._valid_geom_ids] = self._initial_geom_size
            self.model.geom_rgba[self._valid_geom_ids] = self._initial_geom_rgba
            self.model.geom_conaffinity[self._valid_geom_ids] = self._initial_geom_conaffinity
            self.model.geom_contype[self._valid_geom_ids] = self._initial_geom_contype

        self.step_counter = 0
        self.last_head_pos = None
        self.last_head_mat = None
