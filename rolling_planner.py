"""Online one-hour prediction / 7.5-minute execution loop."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

import config
from dynamics.rolling import (
    candidate_cost_numpy,
    distance_km_numpy,
    estimate_route_headwind_numpy,
    energy_aware_speed_limit_numpy,
    make_condition,
    mission_step_limit_numpy,
    plan_expert_controls_numpy,
    rollout_controls_numpy,
    target_from_condition_numpy,
    trajectory_metrics,
)
from models import create_denoiser
from models.diffusion import TrajectoryDiffusion
from models.wind_encoder import SpatioTemporalWindEncoder


@dataclass
class RollingResult:
    start_pos: List[float]
    target_pos: List[float]
    start_time: float
    states: np.ndarray
    controls: np.ndarray
    frames: List[Dict] = field(default_factory=list)
    success: bool = False
    termination: str = "timeout"
    metrics: Dict[str, float] = field(default_factory=dict)


def load_diffusion(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    metadata = checkpoint.get("metadata")
    if metadata is None:
        raise RuntimeError(
            "This is a legacy 32-hour checkpoint. Train/load a rolling-DiT checkpoint instead."
        )
    expected = config.model_metadata()
    mismatch = {key: (metadata.get(key), value) for key, value in expected.items()
                if metadata.get(key) != value}
    if mismatch:
        raise RuntimeError(f"checkpoint/config mismatch: {json.dumps(mismatch, ensure_ascii=False)}")

    denoiser = create_denoiser().to(device)
    wind_encoder = SpatioTemporalWindEncoder().to(device)
    denoiser.load_state_dict(checkpoint["dit"])
    wind_encoder.load_state_dict(checkpoint["wind_encoder"])
    denoiser.eval()
    wind_encoder.eval()
    return TrajectoryDiffusion(denoiser, wind_encoder).to(device).eval()


def _wind_window(wind_frames, start):
    end = start + config.WIND_NUM_FRAMES
    return np.stack([
        frame.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if torch.is_tensor(frame) else np.asarray(frame).squeeze().astype(np.float32)
        for frame in (
            wind_frames[index] if index < len(wind_frames) else wind_frames[-1]
            for index in range(start, end)
        )
    ])


def _shift_previous(controls):
    shifted = np.zeros_like(controls)
    remaining = controls[:, config.EXECUTE_STEPS:]
    shifted[:, :remaining.shape[1]] = remaining
    return shifted


def _sample_local_wind(field, pos_norm):
    """Bilinearly sample one compressed wind frame at a normalized position."""
    height, width = field.shape
    x = float(np.clip(pos_norm[0], 0.0, 1.0)) * (width - 1)
    y = float(np.clip(pos_norm[1], 0.0, 1.0)) * (height - 1)
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
    dx, dy = x - x0, y - y0
    return float(
        field[y0, x0] * (1 - dx) * (1 - dy)
        + field[y0, x1] * dx * (1 - dy)
        + field[y1, x0] * (1 - dx) * dy
        + field[y1, x1] * dx * dy
    )


def _safety_shield_setpoint(current, target_pos, remaining, wind_u, wind_v):
    """Rule-planner-only boundary recovery; never used by DiT inference."""
    cond = make_condition(current, target_pos, remaining)
    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False, dtype=np.float32)
    headings = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    sustainable_ratio = energy_aware_speed_limit_numpy(
        float(current[4]) * 24.0, float(current[5])
    ) / config.MAX_SPEED
    candidates = np.concatenate([
        np.zeros((1, 2), dtype=np.float32),
        headings * (0.75 * sustainable_ratio),
        headings * sustainable_ratio,
        headings,
    ], axis=0)
    target_norm = np.asarray(target_pos, dtype=np.float32) / np.array(
        [config.WIDTH, config.HEIGHT], dtype=np.float32
    )
    best = None
    best_state = None
    best_score = float("inf")
    for setpoint in candidates:
        next_state = rollout_controls_numpy(
            setpoint[:, None], cond, wind_u[None], wind_v[None]
        )[:, 0]
        x_km = float(next_state[0]) * config.WIDTH_KM
        y_km = float(next_state[1]) * config.HEIGHT_KM
        clearance = min(
            x_km, config.WIDTH_KM - x_km,
            y_km, config.HEIGHT_KM - y_km,
        )
        outside = clearance < 0.0
        distance = distance_km_numpy(next_state[:2], target_norm)
        energy_deficit = max(
            0.0, config.ENERGY_SAFE_THRESHOLD + 0.02 - float(next_state[5])
        )
        # Safety dominates; target distance only breaks ties among similarly
        # safe headings.  Among safe actions, avoid spending the final usable
        # battery reserve on an unnecessarily aggressive correction.
        score = (
            (1.0e6 if outside else 0.0)
            + 1.0e5 * energy_deficit
            - clearance
            + 1.0e-3 * distance
        )
        if score < best_score:
            best_score = score
            best = setpoint.astype(np.float32)
            best_state = next_state
    return best, best_state


@torch.no_grad()
def run_rolling_planner(
    diffusion,
    wind_u_frames,
    wind_v_frames,
    init_pos,
    target_pos,
    wind_start=0,
    start_time=config.START_TIME,
    start_energy=config.START_ENERGY,
    guidance_scale=2.0,
    candidates=config.INFERENCE_CANDIDATES,
    sampler="ddpm",
    ddim_steps=50,
    physics_guidance_scale=config.PHYSICS_GUIDANCE_SCALE,
    device=None,
):
    """Run one complete long-duration task under a pre-generated wind sequence."""
    if diffusion is not None:
        device = device or next(diffusion.parameters()).device
    else:
        device = torch.device("cpu")
    current = np.array([
        init_pos[0] / config.WIDTH, init_pos[1] / config.HEIGHT,
        0.0, 0.0, (start_time % 24.0) / 24.0, start_energy,
    ], dtype=np.float32)
    executed_states = [current.copy()]
    executed_controls = []
    frames = []
    # Training always presents a previous-plan tensor (occasionally all zeros).
    # Passing ``None`` on the first online replan skipped the learned projection
    # bias and created an avoidable train/inference mismatch.
    previous_plan = np.zeros((config.ACTION_DIM, config.TRAJ_LEN), dtype=np.float32)
    has_previous_plan = False
    replan_jumps = []
    candidate_diversities = []
    plan_reset_count = 0
    infeasible_plan_count = 0
    distance_history = []
    target_norm = np.array([target_pos[0] / config.WIDTH, target_pos[1] / config.HEIGHT])
    initial_wind_u = _wind_window(wind_u_frames, wind_start)
    initial_wind_v = _wind_window(wind_v_frames, wind_start)
    route_distance_km = distance_km_numpy(current[:2], target_norm)
    initial_headwind_mps = estimate_route_headwind_numpy(
        initial_wind_u, initial_wind_v, current[:2], target_norm
    )
    mission_step_limit = mission_step_limit_numpy(
        route_distance_km, start_energy, initial_headwind_mps
    )
    termination = "timeout"
    success = False
    mission_step = 0

    while mission_step < mission_step_limit:
        frame_start = wind_start + mission_step
        wind_u = _wind_window(wind_u_frames, frame_start)
        wind_v = _wind_window(wind_v_frames, frame_start)
        remaining = (mission_step_limit - mission_step) / mission_step_limit
        cond_np = make_condition(current, target_pos, remaining)
        current_distance = distance_km_numpy(current[:2], target_norm)
        distance_history.append(current_distance)
        stagnating = False
        if len(distance_history) > config.STAGNATION_WINDOW_STEPS:
            recent = distance_history[-(config.STAGNATION_WINDOW_STEPS + 1):]
            progress = recent[0] - recent[-1]
            regression_from_best = recent[-1] - min(recent)
            stagnating = (
                progress < config.STAGNATION_MIN_PROGRESS_KM
                or regression_from_best > config.STAGNATION_MAX_REGRESSION_KM
            )
        planning_previous = (
            np.zeros_like(previous_plan) if stagnating else previous_plan
        )
        if diffusion is not None and stagnating:
            plan_reset_count += 1
        if diffusion is None:
            best_controls, best_states, expert = plan_expert_controls_numpy(
                cond_np, wind_u, wind_v, stochastic=False,
            )
            all_states = expert["states"]
        else:
            cond = torch.from_numpy(cond_np).unsqueeze(0).to(device)
            wind_u_t = torch.from_numpy(wind_u).unsqueeze(0).to(device)
            wind_v_t = torch.from_numpy(wind_v).unsqueeze(0).to(device)
            previous_t = torch.from_numpy(planning_previous).unsqueeze(0).to(device)
            sampled = diffusion.sample_multi(
                cond, wind_u_t, wind_v_t, previous_t,
                guidance_scale=guidance_scale,
                num_samples=candidates,
                sampler=sampler,
                ddim_steps=ddim_steps,
                eta=config.DDIM_ETA,
                physics_guidance_scale=physics_guidance_scale,
            )
            best_controls = sampled["best_controls"][0].cpu().numpy()
            best_states = sampled["best_states"][0].cpu().numpy()
            all_states = sampled["states"][0].cpu().numpy()
            candidate_costs = sampled["costs"][0].cpu().numpy()
        if diffusion is None:
            candidate_costs = np.asarray(expert["scores"], dtype=np.float32)
        if has_previous_plan:
            overlap = config.TRAJ_LEN - config.EXECUTE_STEPS
            replan_jumps.append(float(np.mean(
                (best_controls[:, :overlap] - previous_plan[:, :overlap]) ** 2
            )))
        if all_states.shape[0] > 1:
            scaled_positions = all_states[:, :2].copy()
            scaled_positions[:, 0] *= config.WIDTH_KM
            scaled_positions[:, 1] *= config.HEIGHT_KM
            pair_values = []
            for left in range(all_states.shape[0]):
                for right in range(left + 1, all_states.shape[0]):
                    pair_values.append(float(np.linalg.norm(
                        scaled_positions[left] - scaled_positions[right], axis=0
                    ).mean()))
            candidate_diversities.append(float(np.mean(pair_values)))
        first_segment = best_states[:, :config.EXECUTE_STEPS]
        margin_x = config.BOUNDARY_MARGIN_KM / config.WIDTH_KM
        margin_y = config.BOUNDARY_MARGIN_KM / config.HEIGHT_KM
        feasible = (
            np.isfinite(best_controls).all()
            and np.isfinite(first_segment).all()
            and np.all((best_states[0] >= margin_x) & (best_states[0] <= 1.0 - margin_x))
            and np.all((best_states[1] >= margin_y) & (best_states[1] <= 1.0 - margin_y))
            and np.all(best_states[5] >= config.ENERGY_SAFE_THRESHOLD)
        )
        if diffusion is not None and not feasible:
            infeasible_plan_count += 1
        model_score = candidate_cost_numpy(
            best_states, best_controls, cond_np, wind_u, wind_v
        )

        frames.append({
            "mission_step": mission_step,
            "time_hours": mission_step * config.DT / 3600.0,
            "start_time_hours": start_time % 24.0,
            "clock_time_hours": (
                start_time + mission_step * config.DT / 3600.0
            ) % 24.0,
            "wind_u": wind_u[0],
            "wind_v": wind_v[0],
            "history": np.stack(executed_states, axis=1),
            "prediction": best_states,
            "candidates": all_states,
            "current": current.copy(),
            "energy": float(current[5]),
            "distance_km": distance_km_numpy(current[:2], target_norm),
            "stagnating": stagnating,
            "plan_reset": bool(diffusion is not None and stagnating),
            "plan_feasible": bool(feasible),
            "candidate_costs": candidate_costs.copy(),
            "selection_score": float(model_score["total"]),
        })

        execute_count = min(config.EXECUTE_STEPS, mission_step_limit - mission_step)
        for offset in range(execute_count):
            remaining_step = (mission_step_limit - mission_step) / mission_step_limit
            step_cond = make_condition(current, target_pos, remaining_step)
            step_control = best_controls[:, offset].copy()
            next_state = rollout_controls_numpy(
                step_control[:, None], step_cond,
                wind_u[offset:offset + 1], wind_v[offset:offset + 1],
            )[:, 0]
            margin_x = config.BOUNDARY_SHIELD_MARGIN_KM / config.WIDTH_KM
            margin_y = config.BOUNDARY_SHIELD_MARGIN_KM / config.HEIGHT_KM
            step_safe = (
                margin_x <= next_state[0] <= 1.0 - margin_x
                and margin_y <= next_state[1] <= 1.0 - margin_y
            )
            # Execute the selected DiT output without substituting a rule
            # action. The separate expert baseline may use its own shield.
            if not step_safe and diffusion is None:
                step_control, next_state = _safety_shield_setpoint(
                    current, target_pos, remaining_step, wind_u[offset], wind_v[offset]
                )
                best_controls[:, offset] = step_control

            current = next_state.copy()
            executed_states.append(current)
            executed_controls.append(step_control.copy())
            mission_step += 1

            if distance_km_numpy(current[:2], target_norm) <= config.TARGET_RANGE:
                success = True
                termination = "target"
                break
            if current[5] <= config.ENERGY_SAFE_THRESHOLD:
                termination = "energy"
                break
            if current[0] < 0 or current[0] > 1 or current[1] < 0 or current[1] > 1:
                termination = "boundary"
                break

        previous_plan = _shift_previous(best_controls)
        has_previous_plan = True
        if success or termination != "timeout":
            break

    states_array = np.stack(executed_states, axis=1)
    controls_array = (
        np.stack(executed_controls, axis=1)
        if executed_controls else np.zeros((config.ACTION_DIM, 0), dtype=np.float32)
    )
    metrics = trajectory_metrics(states_array, target_pos)
    metrics.update({
        "success": float(success),
        "mission_hours": mission_step * config.DT / 3600.0,
        "mission_limit_hours": mission_step_limit * config.DT / 3600.0,
        "initial_headwind_mps": initial_headwind_mps,
        "replans": float(len(frames)),
        "mean_replan_control_jump": float(np.mean(replan_jumps)) if replan_jumps else 0.0,
        "mean_candidate_diversity_km": (
            float(np.mean(candidate_diversities)) if candidate_diversities else 0.0
        ),
        "plan_reset_count": float(plan_reset_count),
        "infeasible_plan_count": float(infeasible_plan_count),
        "stagnation_events": float(sum(frame.get("stagnating", False) for frame in frames)),
    })
    if frames:
        final_frame_index = wind_start + mission_step
        final_wind_u = _wind_window(wind_u_frames, final_frame_index)[0]
        final_wind_v = _wind_window(wind_v_frames, final_frame_index)[0]
        frames.append({
            **frames[-1],
            "mission_step": mission_step,
            "time_hours": mission_step * config.DT / 3600.0,
            "clock_time_hours": (
                start_time + mission_step * config.DT / 3600.0
            ) % 24.0,
            "history": states_array,
            "wind_u": final_wind_u,
            "wind_v": final_wind_v,
            "prediction": np.zeros((config.STATE_DIM, 0), dtype=np.float32),
            "candidates": np.zeros((0, config.STATE_DIM, 0), dtype=np.float32),
            "current": current.copy(),
            "energy": float(current[5]),
            "distance_km": distance_km_numpy(current[:2], target_norm),
        })
    return RollingResult(
        start_pos=list(init_pos), target_pos=list(target_pos),
        start_time=float(start_time % 24.0), states=states_array,
        controls=controls_array, frames=frames, success=success,
        termination=termination, metrics=metrics,
    )


def run_rule_planner(*args, **kwargs):
    """Run the multi-route rule expert on the same rolling mission interface."""
    return run_rolling_planner(None, *args, **kwargs)
