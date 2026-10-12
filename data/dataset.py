"""Generate physically consistent one-hour rolling training windows."""
from __future__ import annotations

import json
import math
import os
import random
import sys
from typing import Dict

import numpy as np
import torch

# Keep Matplotlib/font caches out of a read-only home configuration when data
# generation imports legacy dynamics modules that also import plotting code.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffusion_as1_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/diffusion_as1_cache")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from dynamics.airship import ASUnit_low
from dynamics.memory import Memory_wind_field
from dynamics.ppo_policy import PPOGuide
from dynamics.rolling import (
    energy_aware_speed_limit_numpy,
    estimate_route_headwind_numpy,
    make_condition,
    mission_step_limit_numpy,
    plan_expert_controls_numpy,
    route_mode_index,
    rollout_controls_numpy,
    sample_wind_numpy,
    set_route_mode,
)
from dynamics.wind_field import WindGenerator


class RollingWindowDataset(torch.utils.data.Dataset):
    KEYS = (
        "controls", "conditions", "wind_u", "wind_v", "states",
        "valid_mask", "previous_plan", "previous_mask",
    )

    def __init__(self, arrays: Dict[str, np.ndarray], indices: np.ndarray):
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        values = []
        for key in self.KEYS:
            value = self.arrays[key][index]
            if key in {"valid_mask", "previous_mask"}:
                values.append(torch.as_tensor(value, dtype=torch.bool))
            else:
                values.append(torch.as_tensor(value, dtype=torch.float32))
        return tuple(values)


class TrajectoryDataset:
    """Build rolling windows while keeping all windows from one mission together."""

    def __init__(self, num_trajectories=config.NUM_TRAJECTORIES, cache_dir=config.DATA_DIR):
        self.num_trajectories = num_trajectories
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, f"windows_{config.DATA_VERSION}.npz")
        self._arrays = None
        self.mock_args = type("Args", (), {
            "name": "airship", "action_lim": [[-10, 10], [-10, 10]],
            "max_speed": config.MAX_SPEED, "is_action_smooth": True,
            "dt": config.DT, "action_range": config.ACTION_RANGE,
            "pos_config": {
                "width": config.WIDTH, "width_km": config.WIDTH_KM,
                "height": config.HEIGHT, "height_km": config.HEIGHT_KM,
            },
            "target_range": [config.TARGET_RANGE], "target_range_low": [15],
            "target_range_reward": [20], "start_time": config.START_TIME,
            "start_energy": config.START_ENERGY, "is_energy_limit": True,
            "solar_time": config.SOLAR_TIME, "solar_power": config.SOLAR_POWER,
            "max_power": config.MAX_POWER, "default_power": config.DEFAULT_POWER,
            "max_energy": config.MAX_ENERGY,
            "mass": 7001, "Iz": 5.1e6, "Iaz": 5.1e6,
            "m11": 0.11, "m22": 0.83, "drag_coeff": 0.045,
            "yaw_damp": 1.0, "thrust": 1000, "M_control": 6000,
        })()
        self.ppo_guide = None

    def _expert_action(self, airship, target, wind_s=None):
        """Wind-compensated velocity controller used only to create labels."""
        delta_km = np.array([
            (target[0] - airship.pos_state[0]) / config.WIDTH * config.WIDTH_KM,
            (target[1] - airship.pos_state[1]) / config.HEIGHT * config.HEIGHT_KM,
        ], dtype=np.float32)
        distance_km = float(np.linalg.norm(delta_km))
        direction = delta_km / max(distance_km, 1e-6)
        approach_speed = distance_km * 1000.0 / (config.EXPERT_APPROACH_HOURS * 3600.0)
        ground_speed = float(np.clip(approach_speed, 2.0, config.EXPERT_GROUND_SPEED))
        desired_air = direction * ground_speed
        if wind_s is not None:
            # To command a desired ground velocity, compensate the complete
            # estimated wind rather than only a small arbitrary fraction.
            desired_air -= np.asarray(wind_s, dtype=np.float32)

        # Add an inward velocity before the edge is reached.  This creates
        # safe expert labels instead of teaching the model boundary crossings.
        margin_x = config.BOUNDARY_MARGIN_KM / config.WIDTH_KM * config.WIDTH * 2.0
        margin_y = config.BOUNDARY_MARGIN_KM / config.HEIGHT_KM * config.HEIGHT * 2.0
        inward_speed = 6.0
        x, y = airship.pos_state
        if x < margin_x:
            desired_air[0] += inward_speed * (margin_x - x) / margin_x
        elif x > config.WIDTH - margin_x:
            desired_air[0] -= inward_speed * (x - (config.WIDTH - margin_x)) / margin_x
        if y < margin_y:
            desired_air[1] += inward_speed * (margin_y - y) / margin_y
        elif y > config.HEIGHT - margin_y:
            desired_air[1] -= inward_speed * (y - (config.HEIGHT - margin_y)) / margin_y

        near_edge = (
            x < margin_x or x > config.WIDTH - margin_x
            or y < margin_y or y > config.HEIGHT - margin_y
        )
        max_command_speed = energy_aware_speed_limit_numpy(
            airship.time, airship.energy, emergency=near_edge
        )
        norm = float(np.linalg.norm(desired_air))
        if norm > max_command_speed:
            desired_air *= max_command_speed / norm
        action = (desired_air - airship.speed) / config.ACTION_RANGE
        action += np.random.uniform(
            -config.EXPERT_ACTION_NOISE, config.EXPERT_ACTION_NOISE, size=2
        )
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _local_wind(wind_u, wind_v, frame, pos):
        height, width = wind_u[frame].shape[-2:]
        x = min(width - 1, max(0, int(pos[0] / config.RANGE_SCALE)))
        y = min(height - 1, max(0, int(pos[1] / config.RANGE_SCALE)))
        return np.array([
            float(wind_u[frame][0, y, x]), float(wind_v[frame][0, y, x])
        ], dtype=np.float32)

    @staticmethod
    def _lookahead_wind(wind_u, wind_v, frame, pos, target):
        """Sample the complete wind-context interval along a nominal route."""
        route = np.asarray(target, dtype=np.float32) - np.asarray(pos, dtype=np.float32)
        route_km = math.hypot(
            route[0] / config.WIDTH * config.WIDTH_KM,
            route[1] / config.HEIGHT * config.HEIGHT_KM,
        )
        if route_km < 1e-6:
            return TrajectoryDataset._local_wind(wind_u, wind_v, frame, pos)

        sampled = []
        weights = []
        nominal_speed = 8.0
        for offset in range(config.WIND_NUM_FRAMES):
            travel_km = nominal_speed * config.DT * (offset + 1) / 1000.0
            fraction = min(1.0, travel_km / route_km)
            query_pos = np.asarray(pos, dtype=np.float32) + route * fraction
            sampled.append(TrajectoryDataset._local_wind(
                wind_u, wind_v, frame + offset, query_pos
            ))
            weights.append(math.exp(-2.0 * offset / max(config.WIND_NUM_FRAMES - 1, 1)))
        return np.average(np.asarray(sampled), axis=0, weights=np.asarray(weights)).astype(np.float32)

    @staticmethod
    def _shift_plan(plan):
        shifted = np.zeros_like(plan)
        remaining = plan[:, config.EXECUTE_STEPS:]
        shifted[:, :remaining.shape[1]] = remaining
        return shifted

    @staticmethod
    def _details_route_mode(details):
        selected = int(np.asarray(details["selected_index"]).item())
        offset, speed_ratio = details["modes"][selected]
        return route_mode_index(float(offset), float(speed_ratio))

    def _generate_mission(
        self, airship, wind_lists, init_pos, target_pos, wind_start,
        start_time, start_energy,
    ):
        wind_u, wind_v, wind_u_vec, wind_v_vec = wind_lists
        airship.start_time = start_time
        airship.start_energy = start_energy
        initial = airship.reset(init_pos, target_pos).astype(np.float32)
        initial_wind_u = np.stack([
            wind_u[wind_start + offset].squeeze(0).numpy().astype(np.float32)
            for offset in range(config.WIND_NUM_FRAMES)
        ])
        initial_wind_v = np.stack([
            wind_v[wind_start + offset].squeeze(0).numpy().astype(np.float32)
            for offset in range(config.WIND_NUM_FRAMES)
        ])
        target_norm = np.asarray(target_pos, dtype=np.float32) / np.array(
            [config.WIDTH, config.HEIGHT], dtype=np.float32
        )
        route_distance_km = math.hypot(
            (float(initial[0]) - float(target_norm[0])) * config.WIDTH_KM,
            (float(initial[1]) - float(target_norm[1])) * config.HEIGHT_KM,
        )
        initial_headwind_mps = estimate_route_headwind_numpy(
            initial_wind_u, initial_wind_v, initial[:2], target_norm
        )
        mission_step_limit = mission_step_limit_numpy(
            route_distance_km, start_energy, initial_headwind_mps
        )
        states = [initial]
        controls = []
        active_plan = None
        active_mode = 0
        plan_start = 0
        previous_plans = {}
        route_modes = {}
        termination = "timeout"

        for step in range(mission_step_limit):
            frame = wind_start + step
            local_wind = self._local_wind(wind_u, wind_v, frame, airship.pos_state)
            if self.ppo_guide is None:
                if active_plan is None or step % config.EXECUTE_STEPS == 0:
                    previous_plans[step] = (
                        np.zeros((config.ACTION_DIM, config.TRAJ_LEN), dtype=np.float32)
                        if active_plan is None else self._shift_plan(active_plan)
                    )
                    wind_u_window = np.stack([
                        wind_u[frame + offset].squeeze(0).numpy().astype(np.float32)
                        for offset in range(config.WIND_NUM_FRAMES)
                    ])
                    wind_v_window = np.stack([
                        wind_v[frame + offset].squeeze(0).numpy().astype(np.float32)
                        for offset in range(config.WIND_NUM_FRAMES)
                    ])
                    remaining = (mission_step_limit - step) / mission_step_limit
                    cond = make_condition(states[-1], target_pos, remaining)
                    active_plan, _, details = plan_expert_controls_numpy(
                        cond, wind_u_window, wind_v_window, stochastic=True,
                    )
                    active_mode = self._details_route_mode(details)
                    plan_start = step
                route_modes[step] = active_mode
                setpoint = active_plan[:, step - plan_start] * config.MAX_SPEED
                delta = setpoint - np.asarray(airship.speed, dtype=np.float32)
                delta_norm = float(np.linalg.norm(delta))
                if delta_norm > config.ACTION_RANGE:
                    delta *= config.ACTION_RANGE / delta_norm
                action = np.clip(delta / config.ACTION_RANGE, -1.0, 1.0)
                label_control = active_plan[:, step - plan_start].copy()
            else:
                action = self.ppo_guide.get_action(
                    wind_u_vec[frame], wind_v_vec[frame], states[-1], target_pos, noise_scale=0.15
                )
                action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
                label_control = None

            next_state, *_ = airship.step(action, local_wind, target_pos)
            next_state = np.asarray(next_state, dtype=np.float32)

            # Never add an unsafe transition to the expert dataset.
            if not (0.0 <= next_state[0] <= 1.0 and 0.0 <= next_state[1] <= 1.0):
                termination = "boundary"
                break
            if next_state[5] <= config.ENERGY_SAFE_THRESHOLD:
                termination = "energy"
                break
            # The DiT predicts the commanded setpoint, while the low-level
            # controller and physics determine the achieved airspeed.
            controls.append(
                next_state[2:4].copy() if label_control is None else label_control
            )
            states.append(next_state)

            dx_km = (airship.pos_state[0] - target_pos[0]) / config.WIDTH * config.WIDTH_KM
            dy_km = (airship.pos_state[1] - target_pos[1]) / config.HEIGHT * config.HEIGHT_KM
            if math.hypot(dx_km, dy_km) <= config.TARGET_RANGE:
                termination = "target"
                break
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(controls, dtype=np.float32),
            previous_plans,
            route_modes,
            termination,
            mission_step_limit,
        )

    @staticmethod
    def _split_code(mission_id):
        # Deterministic mission-level 80/10/10 split. Every window derived from
        # one mission inherits this code, preventing adjacent-window leakage.
        slot = mission_id % 10
        if slot < round(config.TRAIN_FRACTION * 10):
            return 0
        if slot < round((config.TRAIN_FRACTION + config.VAL_FRACTION) * 10):
            return 1
        return 2

    @staticmethod
    def quality_report(arrays, strict=True):
        """Measure label quality and reject a cache that would waste a full training run."""
        valid = arrays["valid_mask"].astype(bool)
        states = arrays["states"]
        cond = arrays["conditions"]
        lengths = valid.sum(axis=1)
        rows = np.arange(len(lengths))
        last = np.maximum(lengths - 1, 0)
        start = cond[:, :2]
        target = start + cond[:, [config.COND_GOAL_DX, config.COND_GOAL_DY]]
        end = states[rows, last, :2]
        scale = np.array([config.WIDTH_KM, config.HEIGHT_KM], dtype=np.float32)
        start_distance = np.linalg.norm((start - target) * scale, axis=1)
        end_distance = np.linalg.norm((end - target) * scale, axis=1)
        progress = start_distance - end_distance
        multimodal = arrays.get(
            "multimodal", np.zeros(len(lengths), dtype=np.bool_)
        ).astype(bool)
        hard = arrays.get("hard", np.zeros(len(lengths), dtype=np.bool_)).astype(bool)
        bad_previous = arrays.get(
            "bad_previous", np.zeros(len(lengths), dtype=np.bool_)
        ).astype(bool)
        standard = ~multimodal
        previous_mask = arrays["previous_mask"].astype(bool)
        overlap_error = np.square(
            arrays["previous_plan"] - arrays["controls"]
        ).mean(axis=2)
        copied_previous_fraction = float(
            ((overlap_error < 1e-10) & previous_mask).sum()
            / max(previous_mask.sum(), 1)
        )
        outside = (
            (states[:, :, 0] < 0.0) | (states[:, :, 0] > 1.0)
            | (states[:, :, 1] < 0.0) | (states[:, :, 1] > 1.0)
        ) & valid
        report = {
            "windows": int(len(lengths)),
            "missions": int(len(np.unique(arrays["mission_id"]))),
            "cold_start_fraction": float((~arrays["previous_mask"].any(axis=1)).mean()),
            "full_window_fraction": float((lengths == config.TRAJ_LEN).mean()),
            "nonprogress_fraction": float(
                (progress[standard] < config.MIN_WINDOW_PROGRESS_KM).mean()
                if standard.any() else 0.0
            ),
            "multimodal_regression_fraction": float(
                (progress[multimodal] < 0.0).mean() if multimodal.any() else 0.0
            ),
            "boundary_fraction": float(outside.any(axis=1).mean()),
            "mean_progress_km": float(progress.mean()),
            "recovery_fraction": float(arrays.get(
                "recovery", np.zeros(len(lengths), dtype=np.bool_)
            ).mean()),
            "multimodal_fraction": float(multimodal.mean()),
            "hard_fraction": float(hard.mean()),
            "bad_previous_fraction": float(bad_previous.mean()),
            "copied_previous_fraction": copied_previous_fraction,
        }
        split_hard = {}
        for code, name in ((0, "train"), (1, "val"), (2, "test")):
            split_mask = arrays["split"] == code
            split_hard[name] = float(hard[split_mask].mean()) if split_mask.any() else 0.0
        report["split_hard_fraction"] = split_hard
        print(
            "Dataset quality | "
            f"windows={report['windows']} cold_start={report['cold_start_fraction']:.1%} "
            f"full={report['full_window_fraction']:.1%} "
            f"nonprogress={report['nonprogress_fraction']:.2%} "
            f"detour_regression={report['multimodal_regression_fraction']:.1%} "
            f"boundary={report['boundary_fraction']:.2%} "
            f"progress={report['mean_progress_km']:.1f}km "
            f"recovery={report['recovery_fraction']:.1%} "
            f"multimodal={report['multimodal_fraction']:.1%} "
            f"hard={report['hard_fraction']:.1%} "
            f"bad_previous={report['bad_previous_fraction']:.1%} "
            f"copied_previous={report['copied_previous_fraction']:.2%}"
        )
        print(
            "Difficulty by split | "
            + " ".join(f"{name}={value:.1%}" for name, value in split_hard.items())
        )
        if strict:
            failures = []
            if report["windows"] < max(16, report["missions"]):
                failures.append("too few accepted windows")
            if report["cold_start_fraction"] < 0.15:
                failures.append("cold-start plans are underrepresented")
            if report["nonprogress_fraction"] > 0.005:
                failures.append("non-progress expert windows remain")
            if report["boundary_fraction"] > 0.0:
                failures.append("expert windows cross the boundary")
            if report["copied_previous_fraction"] > 0.05:
                failures.append("previous plans appear copied from current labels")
            if report["multimodal_fraction"] < 0.03:
                failures.append("distinct-route expert windows are underrepresented")
            if split_hard["val"] < 0.30 or split_hard["test"] < 0.30:
                failures.append("hard missions are underrepresented outside training")
            if max(split_hard.values()) - min(split_hard.values()) > 0.25:
                failures.append("difficulty distribution differs too much across splits")
            train_mask = arrays["split"] == 0
            if bad_previous[train_mask].mean() < 0.10:
                failures.append("wrong previous-plan correction samples are underrepresented")
            if failures:
                raise RuntimeError("dataset quality gate failed: " + "; ".join(failures))
        return report

    @staticmethod
    def _perturb_condition(condition, split_code):
        if split_code != 0:
            return condition
        result = condition.copy()
        target_x = result[config.COND_X] + result[config.COND_GOAL_DX]
        target_y = result[config.COND_Y] + result[config.COND_GOAL_DY]
        result[config.COND_X] += np.random.normal(0.0, config.STATE_PERTURB_POSITION)
        result[config.COND_Y] += np.random.normal(0.0, config.STATE_PERTURB_POSITION)
        result[config.COND_VX] += np.random.normal(0.0, config.STATE_PERTURB_SPEED)
        result[config.COND_VY] += np.random.normal(0.0, config.STATE_PERTURB_SPEED)
        result[config.COND_ENERGY] = np.clip(
            result[config.COND_ENERGY] + np.random.normal(0.0, config.STATE_PERTURB_ENERGY),
            0.0, 1.0,
        )
        result[config.COND_GOAL_DX] = target_x - result[config.COND_X]
        result[config.COND_GOAL_DY] = target_y - result[config.COND_Y]
        return result.astype(np.float32)

    def _window(
        self, states, controls, wind_u, wind_v, wind_start, start,
        target_pos, split_code, mission_step_limit, prior_plan=None, route_mode=0,
        expert_rank=0, force_relabel=False, hard=False,
    ):
        length = min(config.TRAJ_LEN, max(len(controls) - start, 0))
        if length < config.MIN_WINDOW_STEPS:
            return None
        target_controls = np.zeros((config.TRAJ_LEN, config.ACTION_DIM), dtype=np.float32)
        target_controls[:length] = controls[start:start + length]
        valid_mask = np.ones(config.TRAJ_LEN, dtype=np.bool_)

        remaining = (mission_step_limit - start) / mission_step_limit
        condition = make_condition(
            states[start], target_pos, remaining, route_mode=route_mode
        )

        frame_start = wind_start + start
        frame_indices = range(frame_start, frame_start + config.WIND_NUM_FRAMES)
        wind_u_window = np.stack([
            wind_u[index].squeeze(0).numpy().astype(np.float32) for index in frame_indices
        ])
        wind_v_window = np.stack([
            wind_v[index].squeeze(0).numpy().astype(np.float32) for index in frame_indices
        ])

        # A recovery sample must be relabelled by the expert after perturbing
        # state. Reusing the original control sequence would pair an altered
        # condition with an invalid answer and create train/inference mismatch.
        is_recovery = (
            expert_rank == 0
            and split_code == 0
            and random.random() < config.RECOVERY_WINDOW_RATIO
        )
        if is_recovery:
            recovery_state = states[start].copy()
            recovery_state[:2] += np.random.normal(
                0.0, config.RECOVERY_POSITION_STD, size=2
            ).astype(np.float32)
            recovery_state[2:4] += np.random.normal(
                0.0, config.STATE_PERTURB_SPEED, size=2
            ).astype(np.float32)
            recovery_state[5] -= random.uniform(0.0, config.RECOVERY_ENERGY_DROP_MAX)
            margin = np.array([
                config.BOUNDARY_MARGIN_KM / config.WIDTH_KM,
                config.BOUNDARY_MARGIN_KM / config.HEIGHT_KM,
            ], dtype=np.float32)
            recovery_state[:2] = np.clip(recovery_state[:2], margin, 1.0 - margin)
            recovery_state[2:4] = np.clip(recovery_state[2:4], -1.0, 1.0)
            recovery_state[5] = np.clip(
                recovery_state[5], config.ENERGY_SAFE_THRESHOLD + 0.05, 1.0
            )
            condition = make_condition(recovery_state, target_pos, remaining)
            recovery_controls, recovery_states, details = plan_expert_controls_numpy(
                condition, wind_u_window, wind_v_window, stochastic=True
            )
            condition = set_route_mode(
                condition, self._details_route_mode(details)
            )
            target_controls = recovery_controls.T.astype(np.float32)
            target_states = recovery_states.T.astype(np.float32)
            length = config.TRAJ_LEN
        else:
            target_states = None

        if force_relabel or expert_rank > 0:
            alternate_controls, alternate_states, details = plan_expert_controls_numpy(
                condition, wind_u_window, wind_v_window,
                stochastic=False, selection_rank=expert_rank,
            )
            condition = set_route_mode(
                condition, self._details_route_mode(details)
            )
            target_controls = alternate_controls.T.astype(np.float32)
            target_states = alternate_states.T.astype(np.float32)
            length = config.TRAJ_LEN

        # If the expert reaches the target before one hour, complete the
        # label with wind-compensated station-keeping setpoints.  Masking this
        # tail would leave the model free to accelerate away from the target.
        if not is_recovery and length < config.TRAJ_LEN:
            current_cond = condition.copy()
            if length > 0:
                partial = rollout_controls_numpy(
                    target_controls[:length].T, current_cond,
                    wind_u_window[:length], wind_v_window[:length],
                )
                current_state = partial[:, -1]
                current_cond = make_condition(
                    current_state, target_pos,
                    (mission_step_limit - start - length) / mission_step_limit,
                )
            for offset in range(length, config.TRAJ_LEN):
                pos = current_cond[[config.COND_X, config.COND_Y]]
                target_norm = np.asarray(target_pos, dtype=np.float32) / np.array(
                    [config.WIDTH, config.HEIGHT], dtype=np.float32
                )
                delta_km = (target_norm - pos) * np.array(
                    [config.WIDTH_KM, config.HEIGHT_KM], dtype=np.float32
                )
                correction = delta_km * 1000.0 / 1800.0
                correction_norm = float(np.linalg.norm(correction))
                if correction_norm > 4.0:
                    correction *= 4.0 / correction_norm
                desired_air = correction - np.array([
                    sample_wind_numpy(wind_u_window[offset], pos),
                    sample_wind_numpy(wind_v_window[offset], pos),
                ], dtype=np.float32)
                hour = math.atan2(
                    float(current_cond[config.COND_TIME_SIN]),
                    float(current_cond[config.COND_TIME_COS]),
                ) % (2.0 * math.pi) * 24.0 / (2.0 * math.pi)
                limit = energy_aware_speed_limit_numpy(
                    hour, current_cond[config.COND_ENERGY]
                )
                norm = float(np.linalg.norm(desired_air))
                if norm > limit:
                    desired_air *= limit / norm
                target_controls[offset] = desired_air / config.MAX_SPEED
                next_state = rollout_controls_numpy(
                    target_controls[offset:offset + 1].T, current_cond,
                    wind_u_window[offset:offset + 1], wind_v_window[offset:offset + 1],
                )[:, 0]
                current_cond = make_condition(
                    next_state, target_pos,
                    (mission_step_limit - start - offset - 1) / mission_step_limit,
                )

        if target_states is None:
            target_states = rollout_controls_numpy(
                target_controls.T, condition, wind_u_window, wind_v_window
            ).T

        valid_states = target_states
        if np.any(
            (valid_states[:, 0] < 0.0) | (valid_states[:, 0] > 1.0)
            | (valid_states[:, 1] < 0.0) | (valid_states[:, 1] > 1.0)
            | (valid_states[:, 5] <= config.ENERGY_SAFE_THRESHOLD)
        ):
            return None
        target_norm = np.asarray(target_pos, dtype=np.float32) / np.array(
            [config.WIDTH, config.HEIGHT], dtype=np.float32
        )
        scale_km = np.array([config.WIDTH_KM, config.HEIGHT_KM], dtype=np.float32)
        start_distance = float(np.linalg.norm((condition[:2] - target_norm) * scale_km))
        end_distance = float(np.linalg.norm((valid_states[-1, :2] - target_norm) * scale_km))
        minimum_progress = (
            -config.MULTIMODAL_MAX_REGRESSION_KM
            if expert_rank > 0 else config.MIN_WINDOW_PROGRESS_KM
        )
        if start_distance <= config.TARGET_RANGE or start_distance - end_distance < minimum_progress:
            return None

        previous_target = np.zeros_like(target_controls)
        previous_mask = np.zeros(config.TRAJ_LEN, dtype=np.bool_)
        # The caller supplies the genuinely shifted plan created at the prior
        # replan. It must never be reconstructed from the current target labels.
        if not is_recovery and prior_plan is not None and start >= config.EXECUTE_STEPS:
            previous_target[:] = np.asarray(prior_plan, dtype=np.float32).T
            previous_mask[:config.TRAJ_LEN - config.EXECUTE_STEPS] = True
        is_bad_previous = False
        if (
            split_code == 0
            and previous_mask.any()
            and random.random() < config.BAD_PREVIOUS_PLAN_RATIO
        ):
            # Teach the DiT to reject a stale/wrong inherited route. Rotating
            # the old plan creates the failure pattern seen in closed-loop
            # inference while the target remains a freshly replanned expert.
            corrupted = previous_target.copy()
            transform = random.choice((
                np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32),
                np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32),
                np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float32),
            ))
            corrupted = corrupted @ transform.T
            corrupted += np.random.normal(
                0.0, config.BAD_PREVIOUS_PLAN_NOISE, size=corrupted.shape
            ).astype(np.float32)
            previous_target[:] = np.clip(corrupted, -1.0, 1.0)
            is_bad_previous = True

        result = {
            "controls": target_controls,
            "conditions": condition,
            "wind_u": wind_u_window.astype(np.float16),
            "wind_v": wind_v_window.astype(np.float16),
            "states": target_states,
            "valid_mask": valid_mask,
            "previous_plan": previous_target,
            "previous_mask": previous_mask,
        }
        result["_is_recovery"] = is_recovery
        result["_is_multimodal"] = expert_rank > 0
        result["_is_hard"] = bool(hard)
        result["_is_bad_previous"] = is_bad_previous
        return result

    @staticmethod
    def _route_wind_cosine(wind_u, wind_v, init_pos, target_pos):
        """Classify the two-hour mean wind sampled along a nominal route."""
        start = np.asarray(init_pos, dtype=np.float32) / np.array(
            [config.WIDTH, config.HEIGHT], dtype=np.float32
        )
        target = np.asarray(target_pos, dtype=np.float32) / np.array(
            [config.WIDTH, config.HEIGHT], dtype=np.float32
        )
        route_km = (target - start) * np.array(
            [config.WIDTH_KM, config.HEIGHT_KM], dtype=np.float32
        )
        direction = route_km / max(float(np.linalg.norm(route_km)), 1e-6)
        winds = []
        frame_count = min(config.WIND_NUM_FRAMES, len(wind_u), len(wind_v))
        for frame in range(frame_count):
            fraction = (frame + 1) / max(frame_count, 1)
            position = start + fraction * (target - start)
            winds.append([
                sample_wind_numpy(wind_u[frame], position),
                sample_wind_numpy(wind_v[frame], position),
            ])
        mean_wind = np.mean(np.asarray(winds, dtype=np.float32), axis=0)
        return float(
            mean_wind @ direction / (np.linalg.norm(mean_wind) + 1e-6)
        )

    @staticmethod
    def _sample_start_goal(hard=False, wind_u=None, wind_v=None):
        for _ in range(200):
            init_pos = [
                random.randint(20, config.WIDTH - 20),
                random.randint(20, config.HEIGHT - 20),
            ]
            target_pos = [
                random.randint(30, config.WIDTH - 30),
                random.randint(30, config.HEIGHT - 30),
            ]
            dx_km = (init_pos[0] - target_pos[0]) / config.WIDTH * config.WIDTH_KM
            dy_km = (init_pos[1] - target_pos[1]) / config.HEIGHT * config.HEIGHT_KM
            distance = math.hypot(dx_km, dy_km)
            if hard and config.HARD_ROUTE_MIN_KM <= distance <= config.HARD_ROUTE_MAX_KM:
                if wind_u is None or wind_v is None:
                    return init_pos, target_pos
                cosine = TrajectoryDataset._route_wind_cosine(
                    wind_u, wind_v, init_pos, target_pos
                )
                if cosine < config.HARD_HEADWIND_COSINE:
                    return init_pos, target_pos
            elif not hard and distance >= 80.0:
                return init_pos, target_pos
        if hard:
            return None, None
        return init_pos, target_pos

    def _select_window_starts(
        self, states, controls, wind_u, wind_v, wind_start, target_pos,
        mission_step_limit,
    ):
        """Keep cold-start, adverse-wind, low-energy and one random window."""
        starts = list(range(0, len(controls), config.EXECUTE_STEPS))
        if len(starts) <= config.WINDOWS_PER_MISSION:
            return starts
        target = np.asarray(target_pos, dtype=np.float32)
        ranked = []
        for start in starts[1:]:
            pos = states[start, :2] * np.array([config.WIDTH, config.HEIGHT])
            route = target - pos
            route_norm = float(np.linalg.norm(route))
            direction = route / max(route_norm, 1e-6)
            wind = self._local_wind(
                wind_u, wind_v, wind_start + start, pos
            )
            headwind = max(0.0, -float(wind @ direction))
            risk = (
                2.0 * (1.0 - float(states[start, 5]))
                + headwind / max(config.WIND_MAX_SPEED, 1.0)
                + 0.2 * start / mission_step_limit
            )
            ranked.append((risk, start))
        ranked.sort(reverse=True)
        selected = [0]
        if ranked:
            selected.append(ranked[0][1])
        remaining = [start for start in starts[1:] if start not in selected]
        if remaining and len(selected) < config.WINDOWS_PER_MISSION:
            selected.append(random.choice(remaining))
        for _, start in ranked[1:]:
            if len(selected) >= config.WINDOWS_PER_MISSION:
                break
            if start not in selected:
                selected.append(start)
        return selected

    def generate_all(self, use_ppo=False):
        wind_generator = WindGenerator(
            config.WIDTH, config.HEIGHT,
            config.WIND_MIN_AVE_V, config.WIND_MAX_AVE_V,
            config.WIND_VARY_SPEED, config.WIND_VARY_NUM,
            10, 10, config.WIND_NUM_INP, config.RANGE_SCALE,
        )
        # Small preflight datasets still need enough independent wind
        # sequences to test generalization.  With the old minimum of three,
        # a 96-mission smoke run had only two training winds and could fit the
        # window labels while failing every unseen closed-loop wind field.
        buffer_size = min(config.WIND_BUFFER_SIZE, max(20, self.num_trajectories // 50))
        split_counts = {
            0: max(1, round(buffer_size * config.TRAIN_FRACTION)),
            1: max(1, round(buffer_size * config.VAL_FRACTION)),
        }
        split_counts[2] = max(1, buffer_size - split_counts[0] - split_counts[1])
        wind_buffers = {
            split: Memory_wind_field(count) for split, count in split_counts.items()
        }
        for split, count in split_counts.items():
            for _ in range(count):
                wind_u, wind_v, wind_u_vec, wind_v_vec = wind_generator.generate_wind_vector_list(
                    config.WIND_HOURS
                )
                if not use_ppo:
                    wind_u_vec = wind_v_vec = None
                wind_buffers[split].add_list(wind_u, wind_v, wind_u_vec, wind_v_vec)
        print(
            "Independent wind sequences | "
            f"train={split_counts[0]} val={split_counts[1]} test={split_counts[2]}"
        )

        if use_ppo:
            for directory in ("./checkpoints/ppo_policy", "./ppo_policy"):
                if os.path.exists(os.path.join(directory, "PPO_policy.pth")):
                    self.ppo_guide = PPOGuide(directory, device="cpu")
                    break
        print(
            "Using PPO guide" if self.ppo_guide is not None
            else "Using multi-route energy-aware expert controller"
        )

        airship = ASUnit_low(self.mock_args)
        samples = {key: [] for key in RollingWindowDataset.KEYS}
        split_codes = []
        mission_ids = []
        recovery_flags = []
        multimodal_flags = []
        hard_flags = []
        bad_previous_flags = []
        failure_reasons = {"empty": 0, "boundary": 0, "energy": 0, "timeout": 0, "no_window": 0}

        def store_window(window, split_code, stored_mission_id, force_recovery=False):
            is_recovery = bool(window.pop("_is_recovery")) or force_recovery
            is_multimodal = bool(window.pop("_is_multimodal"))
            is_hard = bool(window.pop("_is_hard"))
            is_bad_previous = bool(window.pop("_is_bad_previous"))
            for key, value in window.items():
                samples[key].append(value)
            split_codes.append(split_code)
            mission_ids.append(stored_mission_id)
            recovery_flags.append(is_recovery)
            multimodal_flags.append(is_multimodal)
            hard_flags.append(is_hard)
            bad_previous_flags.append(is_bad_previous)

        mission_id = 0
        accepted_by_split = {0: 0, 1: 0, 2: 0}
        attempts = 0
        max_attempts = self.num_trajectories * 8
        while mission_id < self.num_trajectories and attempts < max_attempts:
            attempts += 1
            if attempts == 1 or attempts % 50 == 0:
                print(
                    f"Generating successful rolling mission {mission_id}/"
                    f"{self.num_trajectories} (attempt {attempts})"
                )
            split_code = self._split_code(mission_id)
            wind_u, wind_v, wind_u_vec, wind_v_vec = wind_buffers[split_code].get_list()
            required = config.MAX_MISSION_STEPS + config.WIND_NUM_FRAMES + 1
            max_start = len(wind_u) - required
            if max_start < 0:
                raise RuntimeError("generated wind sequence is too short for mission plus planning window")
            wind_start = random.randint(0, max_start)
            # Difficulty and train/val/test assignment use independent cycles.
            # The previous shared mission_id % 10 rule accidentally placed all
            # deliberately hard missions in the training split.
            hard_slots = round(config.HARD_SAMPLE_RATIO * 20)
            hard = (accepted_by_split[split_code] * 7) % 20 < hard_slots
            wind_u_context = np.stack([
                wind_u[wind_start + offset].squeeze(0).numpy().astype(np.float32)
                for offset in range(config.WIND_NUM_FRAMES)
            ])
            wind_v_context = np.stack([
                wind_v[wind_start + offset].squeeze(0).numpy().astype(np.float32)
                for offset in range(config.WIND_NUM_FRAMES)
            ])
            init_pos, target_pos = self._sample_start_goal(
                hard=hard, wind_u=wind_u_context, wind_v=wind_v_context
            )
            if init_pos is None:
                failure_reasons["no_window"] += 1
                continue
            if hard:
                start_time = random.choice((
                    random.uniform(0.0, config.SOLAR_TIME[0]),
                    random.uniform(config.SOLAR_TIME[1], 24.0),
                ))
                start_energy = random.uniform(*config.HARD_START_ENERGY_RANGE)
            else:
                start_time = random.uniform(0.0, 24.0)
                start_energy = random.uniform(*config.TRAIN_START_ENERGY_RANGE)
            (
                states, controls, previous_plans, route_modes,
                termination, mission_step_limit,
            ) = self._generate_mission(
                airship, (wind_u, wind_v, wind_u_vec, wind_v_vec),
                init_pos, target_pos, wind_start, start_time, start_energy,
            )
            if len(controls) == 0:
                failure_reasons["empty"] += 1
                continue
            target_norm = np.asarray(target_pos, dtype=np.float32) / np.array(
                [config.WIDTH, config.HEIGHT], dtype=np.float32
            )
            final_error = np.linalg.norm(
                (states[-1, :2] - target_norm)
                * np.array([config.WIDTH_KM, config.HEIGHT_KM], dtype=np.float32)
            )
            if final_error > config.TARGET_RANGE:
                failure_reasons[termination] = failure_reasons.get(termination, 0) + 1
                # Preserve a difficult state only after the improved expert
                # replans it into a locally safe, progressing target. Raw
                # failure controls are never used as labels.
                if split_code == 0 and len(controls) > 0:
                    failure_starts = sorted({
                        max(0, len(controls) - config.TRAJ_LEN * (index + 1))
                        for index in range(config.FAILURE_RECOVERY_WINDOWS)
                    })
                    for failure_start in failure_starts:
                        recovery_window = self._window(
                            states, controls, wind_u, wind_v, wind_start,
                            failure_start, target_pos, split_code,
                            mission_step_limit,
                            prior_plan=previous_plans.get(failure_start),
                            route_mode=route_modes.get(failure_start, 0),
                            force_relabel=True, hard=True,
                        )
                        if recovery_window is not None:
                            store_window(
                                recovery_window, 0, -attempts,
                                force_recovery=True,
                            )
                continue

            selected_starts = self._select_window_starts(
                states, controls, wind_u, wind_v, wind_start, target_pos,
                mission_step_limit,
            )
            accepted = 0
            high_risk_start = selected_starts[1] if len(selected_starts) > 1 else None
            for start in selected_starts:
                ranks = [0]
                if (
                    split_code == 0
                    and hard
                    and start == high_risk_start
                    and config.MULTIMODAL_HARD_WINDOWS > 0
                ):
                    ranks.extend(range(1, config.MULTIMODAL_HARD_WINDOWS + 1))
                for expert_rank in ranks:
                    window = self._window(
                        states, controls, wind_u, wind_v, wind_start, start, target_pos,
                        split_code, mission_step_limit,
                        prior_plan=previous_plans.get(start),
                        route_mode=route_modes.get(start, 0),
                        expert_rank=expert_rank, hard=hard,
                    )
                    if window is None:
                        continue
                    store_window(window, split_code, mission_id)
                    accepted += 1
                    if accepted >= config.WINDOWS_PER_MISSION:
                        break
                if accepted >= config.WINDOWS_PER_MISSION:
                    break
            if accepted == 0:
                failure_reasons["no_window"] += 1
                continue
            accepted_by_split[split_code] += 1
            mission_id += 1

        if mission_id < self.num_trajectories:
            raise RuntimeError(
                f"only generated {mission_id}/{self.num_trajectories} successful missions "
                f"after {attempts} attempts"
            )
        print(
            f"Accepted {mission_id} successful missions from {attempts} attempts "
            f"({mission_id / max(attempts, 1):.1%})"
        )
        print("Rejected mission reasons: " + json.dumps(failure_reasons, ensure_ascii=False))

        arrays = {key: np.stack(value) for key, value in samples.items()}
        arrays["split"] = np.asarray(split_codes, dtype=np.uint8)
        arrays["mission_id"] = np.asarray(mission_ids, dtype=np.int32)
        arrays["recovery"] = np.asarray(recovery_flags, dtype=np.bool_)
        arrays["multimodal"] = np.asarray(multimodal_flags, dtype=np.bool_)
        arrays["hard"] = np.asarray(hard_flags, dtype=np.bool_)
        arrays["bad_previous"] = np.asarray(bad_previous_flags, dtype=np.bool_)
        arrays["metadata"] = np.asarray(json.dumps(config.model_metadata(), ensure_ascii=False))
        self.quality_report(arrays, strict=True)
        np.savez_compressed(self.cache_file, **arrays)
        self._arrays = arrays
        print(f"Saved {len(split_codes)} rolling windows to {self.cache_file}")
        return arrays

    def load(self):
        if self._arrays is not None:
            return self._arrays
        with np.load(self.cache_file, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        metadata = json.loads(str(arrays["metadata"].item()))
        expected = config.model_metadata()
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"dataset/config mismatch: {json.dumps(mismatches, ensure_ascii=False)}"
            )
        self.quality_report(arrays, strict=True)
        self._arrays = arrays
        return arrays

    def get_dataloader(self, batch_size=config.BATCH_SIZE, split="train"):
        arrays = self.load() if os.path.exists(self.cache_file) else self.generate_all()
        split_value = {"train": 0, "val": 1, "test": 2}[split]
        indices = np.flatnonzero(arrays["split"] == split_value)
        dataset = RollingWindowDataset(arrays, indices)
        sampler = None
        if split == "train":
            weights = np.ones(len(indices), dtype=np.float64)
            weights[arrays["hard"][indices].astype(bool)] *= config.HARD_SAMPLE_WEIGHT
            weights[arrays["recovery"][indices].astype(bool)] *= 1.25
            weights[arrays["bad_previous"][indices].astype(bool)] *= 1.25
            sampler = torch.utils.data.WeightedRandomSampler(
                torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(indices), replacement=True,
            )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=0,
            pin_memory=False,
        )
