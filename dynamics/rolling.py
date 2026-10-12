"""Shared rolling-horizon state, wind and physics utilities.

The DiT predicts normalized airspeed setpoints.  A bounded low-level velocity
controller turns them into increments before applying the same simplified
kinematics and energy model as :mod:`dynamics.airship`.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import config


_SOLAR_COEF = np.polyfit(np.array([5.0, 13.0, 17.0]), np.array([0.0, 18.0, 0.0]), 2)


def set_route_mode(condition: np.ndarray, route_mode: int) -> np.ndarray:
    """Return a condition carrying one explicit route-mode request."""
    result = np.asarray(condition, dtype=np.float32).copy()
    result[config.COND_ROUTE_MODE_START:config.COND_DIM] = 0.0
    index = int(np.clip(route_mode, 0, config.ROUTE_MODE_COUNT - 1))
    result[config.COND_ROUTE_MODE_START + index] = 1.0
    return result


def route_mode_index(offset_km: float, speed_ratio: float) -> int:
    """Map one offline-expert candidate to a stable DiT route-mode class."""
    if speed_ratio <= 0.60:
        return config.ROUTE_MODE_NAMES.index("economy")
    if speed_ratio >= 0.99:
        return config.ROUTE_MODE_NAMES.index("breakthrough")
    if offset_km < -1.0:
        return config.ROUTE_MODE_NAMES.index("left")
    if offset_km > 1.0:
        return config.ROUTE_MODE_NAMES.index("right")
    return config.ROUTE_MODE_NAMES.index("direct")


def make_condition(
    state: np.ndarray,
    target_pos,
    remaining_fraction: float,
    route_mode: int = 0,
) -> np.ndarray:
    """Build the physical rolling condition plus a route-mode request."""
    state = np.asarray(state, dtype=np.float32)
    target_x = float(target_pos[0]) / config.WIDTH
    target_y = float(target_pos[1]) / config.HEIGHT
    hour = float(state[4]) * 24.0
    phase = 2.0 * math.pi * hour / 24.0
    base = np.array([
        state[0], state[1], state[2], state[3],
        math.sin(phase), math.cos(phase), state[5],
        target_x - state[0], target_y - state[1],
        float(np.clip(remaining_fraction, 0.0, 1.0)),
    ], dtype=np.float32)
    condition = np.zeros(config.COND_DIM, dtype=np.float32)
    condition[:config.COND_BASE_DIM] = base
    return set_route_mode(condition, route_mode)


def condition_hour_numpy(cond: np.ndarray) -> float:
    """Recover hour-of-day from cyclic time features."""
    phase = math.atan2(float(cond[config.COND_TIME_SIN]), float(cond[config.COND_TIME_COS]))
    return (phase % (2.0 * math.pi)) * 24.0 / (2.0 * math.pi)


def target_from_condition_numpy(cond: np.ndarray) -> np.ndarray:
    return np.array([
        cond[config.COND_X] + cond[config.COND_GOAL_DX],
        cond[config.COND_Y] + cond[config.COND_GOAL_DY],
    ], dtype=np.float32)


def target_from_condition_torch(cond: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        cond[:, config.COND_X] + cond[:, config.COND_GOAL_DX],
        cond[:, config.COND_Y] + cond[:, config.COND_GOAL_DY],
    ], dim=-1)


def distance_km_numpy(pos_norm: np.ndarray, target_norm: np.ndarray) -> float:
    dx = (float(pos_norm[0]) - float(target_norm[0])) * config.WIDTH_KM
    dy = (float(pos_norm[1]) - float(target_norm[1])) * config.HEIGHT_KM
    return math.hypot(dx, dy)


def distance_km_torch(pos_norm: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    dx = (pos_norm[..., 0] - target_norm[..., 0]) * config.WIDTH_KM
    dy = (pos_norm[..., 1] - target_norm[..., 1]) * config.HEIGHT_KM
    return torch.sqrt(dx.square() + dy.square() + 1e-8)


def solar_power_numpy(hour: float) -> float:
    """Solar input power shared by data generation and rolling inference."""
    if config.SOLAR_TIME[0] < hour < config.SOLAR_TIME[1]:
        return max(0.0, float(np.polyval(_SOLAR_COEF, hour)) * 1.2)
    return 0.0


def _solar_torch(hour: torch.Tensor) -> torch.Tensor:
    a, b, c = [float(v) for v in _SOLAR_COEF]
    power = (a * hour.square() + b * hour + c) * 1.2
    daylight = (hour > config.SOLAR_TIME[0]) & (hour < config.SOLAR_TIME[1])
    return torch.where(daylight, power.clamp_min(0.0), torch.zeros_like(power))


def update_energy_numpy(energy: float, speed_mps: float, hour: float) -> float:
    """Advance the shared battery model by one physical time step."""
    use_power = (
        float(speed_mps) ** 3 * config.PROPULSION_POWER_COEFF
        + config.DEFAULT_POWER
    )
    return float(np.clip(
        float(energy)
        + (solar_power_numpy(float(hour)) - use_power)
        * config.DT / 3600.0 / config.MAX_ENERGY,
        0.0,
        1.0,
    ))


def hours_to_sunrise_numpy(hour: float) -> float:
    """Return hours until the next solar-input interval begins."""
    hour = float(hour) % 24.0
    if config.SOLAR_TIME[0] <= hour < config.EXPERT_ENERGY_CONSERVATION_HOUR:
        return 0.0
    if hour < config.SOLAR_TIME[0]:
        return config.SOLAR_TIME[0] - hour
    return 24.0 - hour + config.SOLAR_TIME[0]


def required_energy_reserve_numpy(hour: float, remaining_distance_km: float) -> float:
    """Dynamic reserve for darkness and the uncompleted part of the route."""
    darkness = (
        hours_to_sunrise_numpy(hour) * config.DEFAULT_POWER / config.MAX_ENERGY
    )
    route = min(0.25, max(0.0, float(remaining_distance_km)) / 1000.0)
    return float(np.clip(
        config.ENERGY_RESERVE_BASE + darkness + route,
        config.ENERGY_SAFE_THRESHOLD,
        config.ENERGY_RESERVE_MAX,
    ))


def required_energy_reserve_torch(hour: torch.Tensor, remaining_distance_km: torch.Tensor):
    """Differentiable counterpart of :func:`required_energy_reserve_numpy`."""
    hour = torch.remainder(hour, 24.0)
    before_sunrise = config.SOLAR_TIME[0] - hour
    after_sunset = 24.0 - hour + config.SOLAR_TIME[0]
    darkness = torch.where(
        hour < config.SOLAR_TIME[0],
        before_sunrise,
        torch.where(hour < config.SOLAR_TIME[1], torch.zeros_like(hour), after_sunset),
    )
    darkness = darkness * config.DEFAULT_POWER / config.MAX_ENERGY
    route = (remaining_distance_km / 1000.0).clamp(0.0, 0.25)
    return (config.ENERGY_RESERVE_BASE + darkness + route).clamp(
        config.ENERGY_SAFE_THRESHOLD, config.ENERGY_RESERVE_MAX
    )


def energy_aware_speed_limit_numpy(hour: float, energy: float, emergency=False) -> float:
    """Return an airspeed that can preserve the energy reserve until daylight."""
    if emergency:
        return config.MAX_SPEED
    hour = float(hour) % 24.0
    if config.SOLAR_TIME[0] <= hour < config.EXPERT_ENERGY_CONSERVATION_HOUR:
        hours_to_sunrise = 0.0
    elif hour < config.SOLAR_TIME[0]:
        hours_to_sunrise = config.SOLAR_TIME[0] - hour
    else:
        hours_to_sunrise = 24.0 - hour + config.SOLAR_TIME[0]
    reserve = max(0.0, float(energy) - config.ENERGY_SAFE_THRESHOLD - 0.05)
    battery_budget_kw = reserve * config.MAX_ENERGY / max(hours_to_sunrise, 1.0)
    propulsion_budget = max(
        0.0, solar_power_numpy(hour) + battery_budget_kw - config.DEFAULT_POWER
    )
    sustainable = min(
        config.MAX_SPEED,
        (propulsion_budget / max(config.PROPULSION_POWER_COEFF, 1e-9)) ** (1.0 / 3.0),
    )
    return float(np.clip(
        sustainable, 0.0, config.MAX_SPEED * config.EXPERT_AIRSPEED_RESERVE
    ))


def sample_wind_numpy(field: np.ndarray, pos_norm: np.ndarray) -> float:
    """Bilinearly sample one compressed wind component at normalized position."""
    h, w = field.shape
    x = float(np.clip(pos_norm[0], 0.0, 1.0)) * (w - 1)
    y = float(np.clip(pos_norm[1], 0.0, 1.0)) * (h - 1)
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0
    return float(
        field[y0, x0] * (1 - dx) * (1 - dy)
        + field[y0, x1] * dx * (1 - dy)
        + field[y1, x0] * (1 - dx) * dy
        + field[y1, x1] * dx * dy
    )


def sample_wind_batch_numpy(field: np.ndarray, pos_norm: np.ndarray) -> np.ndarray:
    """Vectorized bilinear sampling for many candidate positions in one frame."""
    positions = np.asarray(pos_norm, dtype=np.float32)
    h, w = field.shape
    x = np.clip(positions[:, 0], 0.0, 1.0) * (w - 1)
    y = np.clip(positions[:, 1], 0.0, 1.0) * (h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    dx = x - x0
    dy = y - y0
    return (
        field[y0, x0] * (1.0 - dx) * (1.0 - dy)
        + field[y0, x1] * dx * (1.0 - dy)
        + field[y1, x0] * (1.0 - dx) * dy
        + field[y1, x1] * dx * dy
    ).astype(np.float32)


def estimate_route_headwind_numpy(
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    start_norm: np.ndarray,
    target_norm: np.ndarray,
) -> float:
    """Estimate adverse wind over the available near-term route context."""
    start = np.asarray(start_norm, dtype=np.float32)
    target = np.asarray(target_norm, dtype=np.float32)
    scale = np.asarray([config.WIDTH_KM, config.HEIGHT_KM], dtype=np.float32)
    route_km = (target - start) * scale
    route_distance = float(np.linalg.norm(route_km))
    if route_distance < 1e-6:
        return 0.0
    direction = route_km / route_distance
    steps = min(config.WIND_NUM_FRAMES, len(wind_u), len(wind_v))
    values = []
    for offset in range(steps):
        travel_km = config.EXPERT_GROUND_SPEED * config.DT * (offset + 1) / 1000.0
        fraction = min(1.0, travel_km / route_distance)
        position = start + (target - start) * fraction
        wind = np.asarray([
            sample_wind_numpy(wind_u[offset], position),
            sample_wind_numpy(wind_v[offset], position),
        ], dtype=np.float32)
        values.append(max(0.0, -float(wind @ direction)))
    return float(np.percentile(values, 75.0)) if values else 0.0


def mission_time_limit_hours(
    route_distance_km: float,
    start_energy: float,
    initial_headwind_mps: float,
) -> float:
    """Compute a deadline from distance, initial energy and near-term wind."""
    ideal_hours = float(route_distance_km) / max(
        config.EXPERT_GROUND_SPEED * 3.6, 1e-6
    )
    headwind_fraction = np.clip(
        float(initial_headwind_mps) / max(config.EXPERT_GROUND_SPEED, 1e-6),
        0.0, 1.0,
    )
    low_energy_fraction = np.clip(
        (0.80 - float(start_energy)) / 0.25, 0.0, 1.0
    )
    hours = (
        config.MISSION_TIME_FIXED_HOURS
        + config.MISSION_TIME_DISTANCE_FACTOR * ideal_hours
        + config.MISSION_TIME_HEADWIND_BUFFER_HOURS * headwind_fraction
        + config.MISSION_TIME_LOW_ENERGY_BUFFER_HOURS * low_energy_fraction
    )
    hours = float(np.clip(
        hours, config.MIN_MISSION_HOURS, config.MAX_MISSION_HOURS
    ))
    return math.ceil(hours * 3600.0 / config.DT) * config.DT / 3600.0


def mission_step_limit_numpy(
    route_distance_km: float,
    start_energy: float,
    initial_headwind_mps: float,
) -> int:
    return int(round(
        mission_time_limit_hours(
            route_distance_km, start_energy, initial_headwind_mps
        ) * 3600.0 / config.DT
    ))


def _sample_wind_torch(field: torch.Tensor, pos_norm: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample ``field`` (B,H,W) at ``pos_norm`` (B,2)."""
    grid = pos_norm.clamp(0.0, 1.0).mul(2.0).sub(1.0).view(-1, 1, 1, 2)
    sampled = F.grid_sample(
        field.unsqueeze(1), grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return sampled[:, 0, 0, 0]


def _bounded_velocity_delta_numpy(setpoint: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """Limit total velocity-vector change, rather than each axis independently."""
    delta = np.asarray(setpoint, dtype=np.float32) - np.asarray(speed, dtype=np.float32)
    norm = float(np.linalg.norm(delta))
    if norm > config.ACTION_RANGE:
        delta *= config.ACTION_RANGE / norm
    return delta


def _bounded_velocity_delta_torch(setpoint: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
    delta = setpoint - speed
    norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    return delta * (config.ACTION_RANGE / norm.clamp_min(config.ACTION_RANGE))


def rollout_controls_numpy(
    controls: np.ndarray,
    cond: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
) -> np.ndarray:
    """Roll a normalized control sequence into states, returning shape (6,T)."""
    controls = np.asarray(controls, dtype=np.float32)
    if controls.shape[0] != config.ACTION_DIM:
        controls = controls.T
    steps = controls.shape[1]
    pos = np.array([cond[config.COND_X], cond[config.COND_Y]], dtype=np.float32)
    speed = np.array([cond[config.COND_VX], cond[config.COND_VY]], dtype=np.float32) * config.MAX_SPEED
    energy = float(cond[config.COND_ENERGY])
    hour = condition_hour_numpy(cond)
    states = np.zeros((config.STATE_DIM, steps), dtype=np.float32)

    for i in range(steps):
        setpoint = np.clip(controls[:, i], -1.0, 1.0) * config.MAX_SPEED
        velocity_delta = _bounded_velocity_delta_numpy(setpoint, speed)
        speed = np.clip(speed + velocity_delta, -config.MAX_SPEED, config.MAX_SPEED)
        speed_norm = float(np.linalg.norm(speed))
        if speed_norm > config.MAX_SPEED:
            speed *= config.MAX_SPEED / speed_norm
            speed_norm = config.MAX_SPEED

        hour = (hour + config.DT / 3600.0) % 24.0
        energy = update_energy_numpy(energy, speed_norm, hour)
        if energy < config.ENERGY_SAFE_THRESHOLD:
            speed[:] = 0.0

        wind = np.array([
            sample_wind_numpy(wind_u[i], pos),
            sample_wind_numpy(wind_v[i], pos),
        ], dtype=np.float32)
        ground_speed = speed + wind
        pos[0] += config.DT * ground_speed[0] / 1000.0 / config.WIDTH_KM
        pos[1] += config.DT * ground_speed[1] / 1000.0 / config.HEIGHT_KM

        states[:, i] = [
            pos[0], pos[1], speed[0] / config.MAX_SPEED, speed[1] / config.MAX_SPEED,
            hour / 24.0, energy,
        ]
    return states


def rollout_controls_torch(
    controls: torch.Tensor,
    cond: torch.Tensor,
    wind_u: torch.Tensor,
    wind_v: torch.Tensor,
) -> torch.Tensor:
    """Differentiable batched rollout, returning states with shape (B,6,T)."""
    batch, _, steps = controls.shape
    pos = cond[:, [config.COND_X, config.COND_Y]].clone()
    speed = cond[:, [config.COND_VX, config.COND_VY]] * config.MAX_SPEED
    energy = cond[:, config.COND_ENERGY].clone()
    phase = torch.atan2(cond[:, config.COND_TIME_SIN], cond[:, config.COND_TIME_COS])
    hour = torch.remainder(phase, 2.0 * math.pi) * 24.0 / (2.0 * math.pi)
    states = []

    for i in range(steps):
        setpoint = controls[:, :, i].clamp(-1.0, 1.0) * config.MAX_SPEED
        velocity_delta = _bounded_velocity_delta_torch(setpoint, speed)
        speed = (speed + velocity_delta).clamp(-config.MAX_SPEED, config.MAX_SPEED)
        speed_norm = torch.linalg.vector_norm(speed, dim=-1)
        scale = (config.MAX_SPEED / speed_norm.clamp_min(config.MAX_SPEED)).unsqueeze(-1)
        speed = speed * scale
        speed_norm = torch.linalg.vector_norm(speed, dim=-1)

        hour = torch.remainder(hour + config.DT / 3600.0, 24.0)
        use_power = (
            speed_norm.pow(3) * config.PROPULSION_POWER_COEFF
            + config.DEFAULT_POWER
        )
        energy = (
            energy + (_solar_torch(hour) - use_power) * config.DT / 3600.0 / config.MAX_ENERGY
        ).clamp(0.0, 1.0)
        speed = torch.where(
            (energy < config.ENERGY_SAFE_THRESHOLD).unsqueeze(-1), torch.zeros_like(speed), speed
        )

        wind = torch.stack([
            _sample_wind_torch(wind_u[:, i], pos),
            _sample_wind_torch(wind_v[:, i], pos),
        ], dim=-1)
        ground_speed = speed + wind
        pos = torch.stack([
            pos[:, 0] + config.DT * ground_speed[:, 0] / 1000.0 / config.WIDTH_KM,
            pos[:, 1] + config.DT * ground_speed[:, 1] / 1000.0 / config.HEIGHT_KM,
        ], dim=-1)

        states.append(torch.stack([
            pos[:, 0], pos[:, 1], speed[:, 0] / config.MAX_SPEED,
            speed[:, 1] / config.MAX_SPEED, hour / 24.0, energy,
        ], dim=1))
    return torch.stack(states, dim=-1).view(batch, config.STATE_DIM, steps)


def candidate_cost_numpy(
    states: np.ndarray,
    controls: np.ndarray,
    cond: np.ndarray,
    wind_u: Optional[np.ndarray] = None,
    wind_v: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Physical route score used for expert labels and DiT-candidate ranking."""
    target = target_from_condition_numpy(cond)
    start = cond[[config.COND_X, config.COND_Y]]
    start_distance = distance_km_numpy(start, target)
    end_distance = distance_km_numpy(states[:2, -1], target)
    domain = max(config.WIDTH_KM, config.HEIGHT_KM)

    x_km = states[0] * config.WIDTH_KM
    y_km = states[1] * config.HEIGHT_KM
    margin = config.BOUNDARY_MARGIN_KM
    edge_risk = (
        np.maximum(margin - x_km, 0.0)
        + np.maximum(margin - (config.WIDTH_KM - x_km), 0.0)
        + np.maximum(margin - y_km, 0.0)
        + np.maximum(margin - (config.HEIGHT_KM - y_km), 0.0)
    ) / margin
    boundary = float(np.max(edge_risk))

    terminal_hour = float(states[4, -1]) * 24.0
    reserve = required_energy_reserve_numpy(terminal_hour, end_distance)
    threshold_deficit = float(np.maximum(
        config.ENERGY_SAFE_THRESHOLD - states[5], 0.0
    ).mean())
    reserve_deficit = max(0.0, reserve - float(states[5, -1]))
    energy = threshold_deficit + reserve_deficit

    distances = np.array([
        distance_km_numpy(states[:2, step], target)
        for step in range(states.shape[1])
    ], dtype=np.float32)
    distance_series = np.concatenate([[start_distance], distances])
    regression = float(np.maximum(np.diff(distance_series), 0.0).mean() / domain)
    early_index = min(config.EXECUTE_STEPS, states.shape[1]) - 1
    early_regression = max(0.0, float(distances[early_index]) - start_distance) / domain

    progress_km = max(0.0, start_distance - end_distance)
    energy_spent = max(0.0, float(cond[config.COND_ENERGY]) - float(states[5, -1]))
    efficiency = min(5.0, energy_spent / max(progress_km / 100.0, 0.05))
    smooth = float(np.square(np.diff(controls, axis=1)).mean()) if controls.shape[1] > 1 else 0.0
    trend_start = max(0, len(distances) - 4)
    trend = max(0.0, float(distances[-1] - distances[trend_start])) / domain
    stagnation = max(0.0, config.STAGNATION_MIN_PROGRESS_KM - progress_km) / domain
    closing = 0.0
    if wind_u is not None and wind_v is not None:
        final_index = min(states.shape[1] - 1, len(wind_u) - 1)
        route = (target - states[:2, -1]) * np.asarray(
            [config.WIDTH_KM, config.HEIGHT_KM], dtype=np.float32
        )
        direction = route / max(float(np.linalg.norm(route)), 1e-6)
        wind = np.asarray([
            sample_wind_numpy(wind_u[final_index], states[:2, -1]),
            sample_wind_numpy(wind_v[final_index], states[:2, -1]),
        ], dtype=np.float32)
        ground = states[2:4, -1] * config.MAX_SPEED + wind
        closing = max(0.0, 2.0 - float(ground @ direction)) / config.MAX_SPEED
    total = (
        config.SCORE_PROGRESS * end_distance / domain
        + config.SCORE_BOUNDARY * boundary
        + config.SCORE_ENERGY * energy
        + config.SCORE_SMOOTH * smooth
        + config.SCORE_REGRESSION * regression
        + config.SCORE_EARLY_PROGRESS * early_regression
        + config.SCORE_EFFICIENCY * efficiency
        + config.SCORE_TREND * trend
        + config.SCORE_CLOSING * closing
        + config.SCORE_STAGNATION * stagnation
    )
    return {
        "total": float(total),
        "start_distance_km": float(start_distance),
        "end_distance_km": float(end_distance),
        "progress_km": float(start_distance - end_distance),
        "boundary": boundary,
        "energy": float(energy),
        "required_reserve": reserve,
        "efficiency": float(efficiency),
        "regression": regression,
        "early_regression": early_regression,
        "smooth": smooth,
        "trend": trend,
        "closing": closing,
        "stagnation": stagnation,
    }


def _expert_plan_to_waypoint_numpy(
    cond: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    waypoint_norm: np.ndarray,
    speed_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create one two-hour expert candidate through an intermediate waypoint."""
    controls, states = _expert_plans_batch_numpy(
        cond, wind_u, wind_v,
        np.asarray(waypoint_norm, dtype=np.float32)[None],
        np.asarray([speed_ratio], dtype=np.float32),
    )
    return controls[0], states[0]


def _expert_plans_batch_numpy(
    cond: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    waypoints: np.ndarray,
    speed_ratios: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate every route/speed candidate together using vectorized NumPy."""
    count = len(waypoints)
    steps = min(config.EXPERT_EVAL_STEPS, wind_u.shape[0], wind_v.shape[0])
    if steps < config.TRAJ_LEN:
        raise ValueError("expert wind context is shorter than the control horizon")
    controls = np.zeros((count, config.ACTION_DIM, steps), dtype=np.float32)
    states = np.zeros((count, config.STATE_DIM, steps), dtype=np.float32)
    target = target_from_condition_numpy(cond)
    scale_km = np.array([config.WIDTH_KM, config.HEIGHT_KM], dtype=np.float32)
    margin = np.array([
        config.BOUNDARY_MARGIN_KM / config.WIDTH_KM,
        config.BOUNDARY_MARGIN_KM / config.HEIGHT_KM,
    ], dtype=np.float32)
    pos = np.repeat(
        cond[[config.COND_X, config.COND_Y]][None], count, axis=0
    ).astype(np.float32)
    speed = np.repeat(
        (cond[[config.COND_VX, config.COND_VY]] * config.MAX_SPEED)[None],
        count,
        axis=0,
    ).astype(np.float32)
    energy = np.full(count, cond[config.COND_ENERGY], dtype=np.float32)
    hour = condition_hour_numpy(cond)

    for step in range(steps):
        waypoint_delta = (waypoints - pos) * scale_km
        waypoint_distance = np.linalg.norm(waypoint_delta, axis=1)
        destination = np.where(
            (waypoint_distance < 15.0)[:, None], target[None], waypoints
        )
        delta_km = (destination - pos) * scale_km
        distance = np.linalg.norm(delta_km, axis=1)
        direction = delta_km / np.maximum(distance[:, None], 1e-6)
        final_distance = np.linalg.norm((target[None] - pos) * scale_km, axis=1)
        approach_speed = final_distance * 1000.0 / (
            config.EXPERT_APPROACH_HOURS * 3600.0
        )
        moving = speed_ratios > 0.0
        ground_speed = np.where(
            moving,
            np.minimum(
                np.maximum(approach_speed, 1.0),
                config.EXPERT_GROUND_SPEED * speed_ratios,
            ),
            0.0,
        )
        local_wind = np.stack([
            sample_wind_batch_numpy(wind_u[step], pos),
            sample_wind_batch_numpy(wind_v[step], pos),
        ], axis=1)
        desired_air = direction * ground_speed[:, None] - local_wind
        for axis in range(2):
            low = pos[:, axis] < 2.0 * margin[axis]
            high = pos[:, axis] > 1.0 - 2.0 * margin[axis]
            desired_air[low, axis] += 6.0 * (
                2.0 * margin[axis] - pos[low, axis]
            ) / (2.0 * margin[axis])
            desired_air[high, axis] -= 6.0 * (
                pos[high, axis] - (1.0 - 2.0 * margin[axis])
            ) / (2.0 * margin[axis])
        if config.SOLAR_TIME[0] <= hour < config.EXPERT_ENERGY_CONSERVATION_HOUR:
            hours_to_sunrise = 0.0
        elif hour < config.SOLAR_TIME[0]:
            hours_to_sunrise = config.SOLAR_TIME[0] - hour
        else:
            hours_to_sunrise = 24.0 - hour + config.SOLAR_TIME[0]
        battery_budget = (
            np.maximum(energy - config.ENERGY_SAFE_THRESHOLD - 0.05, 0.0)
            * config.MAX_ENERGY / max(hours_to_sunrise, 1.0)
        )
        propulsion_budget = np.maximum(
            solar_power_numpy(hour) + battery_budget - config.DEFAULT_POWER, 0.0
        )
        sustainable = np.minimum(
            config.MAX_SPEED,
            (
                propulsion_budget
                / max(config.PROPULSION_POWER_COEFF, 1e-9)
            ) ** (1.0 / 3.0),
        )
        speed_limit = np.clip(
            sustainable, 0.0,
            config.MAX_SPEED * config.EXPERT_AIRSPEED_RESERVE,
        )
        headwind = np.maximum(0.0, -(local_wind * direction).sum(axis=1))
        daylight = config.SOLAR_TIME[0] <= hour < config.SOLAR_TIME[1]
        breakthrough = (
            (speed_ratios >= 0.99)
            & (headwind >= config.EXPERT_STRONG_HEADWIND_MPS)
            & (energy >= config.EXPERT_BREAKTHROUGH_ENERGY)
            & (
                daylight
                | (energy >= 0.60)
                | (final_distance <= config.EXPERT_FINISH_DISTANCE_KM)
            )
        )
        speed_limit[breakthrough] = config.MAX_SPEED
        desired_norm = np.linalg.norm(desired_air, axis=1)
        desired_air *= np.minimum(
            1.0, speed_limit / np.maximum(desired_norm, 1e-6)
        )[:, None]
        controls[:, :, step] = desired_air / config.MAX_SPEED

        setpoint = np.clip(controls[:, :, step], -1.0, 1.0) * config.MAX_SPEED
        velocity_delta = setpoint - speed
        delta_norm = np.linalg.norm(velocity_delta, axis=1)
        velocity_delta *= np.minimum(
            1.0, config.ACTION_RANGE / np.maximum(delta_norm, 1e-6)
        )[:, None]
        speed = np.clip(
            speed + velocity_delta, -config.MAX_SPEED, config.MAX_SPEED
        )
        speed_norm = np.linalg.norm(speed, axis=1)
        speed *= np.minimum(
            1.0, config.MAX_SPEED / np.maximum(speed_norm, 1e-6)
        )[:, None]
        speed_norm = np.linalg.norm(speed, axis=1)

        hour = (hour + config.DT / 3600.0) % 24.0
        use_power = (
            speed_norm ** 3 * config.PROPULSION_POWER_COEFF
            + config.DEFAULT_POWER
        )
        energy = np.clip(
            energy
            + (solar_power_numpy(hour) - use_power)
            * config.DT / 3600.0 / config.MAX_ENERGY,
            0.0,
            1.0,
        )
        speed[energy < config.ENERGY_SAFE_THRESHOLD] = 0.0

        ground_speed_vector = speed + local_wind
        pos[:, 0] += (
            config.DT * ground_speed_vector[:, 0] / 1000.0 / config.WIDTH_KM
        )
        pos[:, 1] += (
            config.DT * ground_speed_vector[:, 1] / 1000.0 / config.HEIGHT_KM
        )
        states[:, :, step] = np.stack([
            pos[:, 0], pos[:, 1],
            speed[:, 0] / config.MAX_SPEED,
            speed[:, 1] / config.MAX_SPEED,
            np.full(count, hour / 24.0, dtype=np.float32),
            energy,
        ], axis=1)
    return controls, states


def expert_candidate_cost_numpy(
    states: np.ndarray,
    controls: np.ndarray,
    cond: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
) -> Dict[str, float]:
    """Longer-horizon score used only to select expert demonstrations."""
    result = candidate_cost_numpy(states, controls, cond)
    target = target_from_condition_numpy(cond)
    domain = max(config.WIDTH_KM, config.HEIGHT_KM)
    distances = np.asarray([
        distance_km_numpy(states[:2, step], target)
        for step in range(states.shape[1])
    ], dtype=np.float32)
    trend_start = max(0, len(distances) - 4)
    trend = max(0.0, float(distances[-1] - distances[trend_start])) / domain

    dx = np.diff(np.concatenate([[cond[config.COND_X]], states[0]])) * config.WIDTH_KM
    dy = np.diff(np.concatenate([[cond[config.COND_Y]], states[1]])) * config.HEIGHT_KM
    path = float(np.sqrt(dx * dx + dy * dy).sum()) / domain
    headwind_values = []
    closing_values = []
    for step in range(states.shape[1]):
        pos = states[:2, step]
        route = (target - pos) * np.asarray(
            [config.WIDTH_KM, config.HEIGHT_KM], dtype=np.float32
        )
        direction = route / max(float(np.linalg.norm(route)), 1e-6)
        wind = np.asarray([
            sample_wind_numpy(wind_u[step], pos),
            sample_wind_numpy(wind_v[step], pos),
        ], dtype=np.float32)
        ground = states[2:4, step] * config.MAX_SPEED + wind
        headwind_values.append(max(0.0, -float(wind @ direction)) / config.MAX_SPEED)
        closing_values.append(float(ground @ direction))
    headwind = float(np.mean(headwind_values))
    closing = max(0.0, 2.0 - float(np.mean(closing_values[-3:]))) / config.MAX_SPEED
    survival = max(
        0.0,
        result["required_reserve"] + config.EXPERT_NIGHT_RESERVE_MARGIN
        - float(states[5, -1]),
    )
    result.update({
        "trend": trend, "path": path, "headwind": headwind,
        "closing": closing, "survival": survival,
    })
    result["total"] = float(
        result["total"]
        + config.EXPERT_SCORE_TREND * trend
        + config.EXPERT_SCORE_HEADWIND * headwind
        + config.EXPERT_SCORE_CLOSING * closing
        + config.EXPERT_SCORE_PATH * path
        + config.EXPERT_SCORE_SURVIVAL * survival
    )
    return result


def expert_candidate_costs_batch_numpy(
    states: np.ndarray,
    controls: np.ndarray,
    cond: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
) -> np.ndarray:
    """Vectorized counterpart used while ranking all expert candidates."""
    count, _, steps = states.shape
    target = target_from_condition_numpy(cond)
    scale = np.asarray([config.WIDTH_KM, config.HEIGHT_KM], dtype=np.float32)
    domain = max(config.WIDTH_KM, config.HEIGHT_KM)
    start = cond[[config.COND_X, config.COND_Y]]
    start_distance = distance_km_numpy(start, target)
    distances = np.linalg.norm((states[:, :2].transpose(0, 2, 1) - target) * scale, axis=2)
    end_distance = distances[:, -1]

    x_km = states[:, 0] * config.WIDTH_KM
    y_km = states[:, 1] * config.HEIGHT_KM
    margin = config.BOUNDARY_MARGIN_KM
    boundary = np.max(
        (
            np.maximum(margin - x_km, 0.0)
            + np.maximum(margin - (config.WIDTH_KM - x_km), 0.0)
            + np.maximum(margin - y_km, 0.0)
            + np.maximum(margin - (config.HEIGHT_KM - y_km), 0.0)
        ) / margin,
        axis=1,
    )
    reserve = np.asarray([
        required_energy_reserve_numpy(float(states[i, 4, -1]) * 24.0, end_distance[i])
        for i in range(count)
    ], dtype=np.float32)
    threshold_deficit = np.maximum(
        config.ENERGY_SAFE_THRESHOLD - states[:, 5], 0.0
    ).mean(axis=1)
    reserve_deficit = np.maximum(reserve - states[:, 5, -1], 0.0)
    energy = threshold_deficit + reserve_deficit
    distance_series = np.concatenate([
        np.full((count, 1), start_distance, dtype=np.float32), distances
    ], axis=1)
    regression = np.maximum(np.diff(distance_series, axis=1), 0.0).mean(axis=1) / domain
    early = np.maximum(distances[:, config.EXECUTE_STEPS - 1] - start_distance, 0.0) / domain
    progress = np.maximum(start_distance - end_distance, 0.0)
    energy_spent = np.maximum(cond[config.COND_ENERGY] - states[:, 5, -1], 0.0)
    efficiency = np.minimum(5.0, energy_spent / np.maximum(progress / 100.0, 0.05))
    smooth = np.square(np.diff(controls, axis=2)).mean(axis=(1, 2))
    trend_start = max(0, steps - 4)
    trend = np.maximum(distances[:, -1] - distances[:, trend_start], 0.0) / domain
    stagnation = np.maximum(config.STAGNATION_MIN_PROGRESS_KM - progress, 0.0) / domain

    headwind = np.zeros(count, dtype=np.float32)
    closing_tail = []
    for step in range(steps):
        pos = states[:, :2, step]
        route = (target[None] - pos) * scale
        direction = route / np.maximum(np.linalg.norm(route, axis=1)[:, None], 1e-6)
        local_wind = np.stack([
            sample_wind_batch_numpy(wind_u[step], pos),
            sample_wind_batch_numpy(wind_v[step], pos),
        ], axis=1)
        headwind += np.maximum(0.0, -(local_wind * direction).sum(axis=1)) / config.MAX_SPEED
        if step >= steps - 3:
            ground = states[:, 2:4, step] * config.MAX_SPEED + local_wind
            closing_tail.append((ground * direction).sum(axis=1))
    headwind /= steps
    closing = np.maximum(0.0, 2.0 - np.mean(closing_tail, axis=0)) / config.MAX_SPEED

    dx = np.diff(np.concatenate([
        np.full((count, 1), cond[config.COND_X], dtype=np.float32), states[:, 0]
    ], axis=1), axis=1) * config.WIDTH_KM
    dy = np.diff(np.concatenate([
        np.full((count, 1), cond[config.COND_Y], dtype=np.float32), states[:, 1]
    ], axis=1), axis=1) * config.HEIGHT_KM
    path = np.sqrt(dx * dx + dy * dy).sum(axis=1) / domain
    survival = np.maximum(
        reserve + config.EXPERT_NIGHT_RESERVE_MARGIN - states[:, 5, -1], 0.0
    )
    return (
        config.SCORE_PROGRESS * end_distance / domain
        + config.SCORE_BOUNDARY * boundary
        + config.SCORE_ENERGY * energy
        + config.SCORE_SMOOTH * smooth
        + config.SCORE_REGRESSION * regression
        + config.SCORE_EARLY_PROGRESS * early
        + config.SCORE_EFFICIENCY * efficiency
        + config.SCORE_TREND * trend
        + config.SCORE_STAGNATION * stagnation
        + config.EXPERT_SCORE_TREND * trend
        + config.EXPERT_SCORE_HEADWIND * headwind
        + config.EXPERT_SCORE_CLOSING * closing
        + config.EXPERT_SCORE_PATH * path
        + config.EXPERT_SCORE_SURVIVAL * survival
    ).astype(np.float32)


def plan_expert_controls_numpy(
    cond: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    stochastic: bool = False,
    selection_rank: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Compare direct and lateral routes and return an energy-aware expert plan."""
    start = np.asarray(cond[[config.COND_X, config.COND_Y]], dtype=np.float32)
    target = target_from_condition_numpy(cond)
    scale_km = np.array([config.WIDTH_KM, config.HEIGHT_KM], dtype=np.float32)
    route_km = (target - start) * scale_km
    route_distance = float(np.linalg.norm(route_km))
    direction = route_km / max(route_distance, 1e-6)
    lateral = np.array([-direction[1], direction[0]], dtype=np.float32)
    lookahead = min(config.EXPERT_LOOKAHEAD_KM, max(20.0, route_distance * 0.65))
    margin = np.array([
        config.EXPERT_WAYPOINT_MARGIN_KM / config.WIDTH_KM,
        config.EXPERT_WAYPOINT_MARGIN_KM / config.HEIGHT_KM,
    ], dtype=np.float32)

    waypoints = []
    speed_ratios = []
    modes = []
    for requested_offset in config.EXPERT_ROUTE_OFFSETS_KM:
        offset = float(np.clip(
            requested_offset,
            -max(15.0, route_distance * 0.60),
            max(15.0, route_distance * 0.60),
        ))
        waypoint_km = start * scale_km + direction * lookahead + lateral * offset
        waypoint = np.clip(waypoint_km / scale_km, margin, 1.0 - margin)
        for speed_ratio in config.EXPERT_SPEED_RATIOS:
            if speed_ratio <= 0.0 and requested_offset != 0.0:
                continue
            waypoints.append(waypoint)
            speed_ratios.append(float(speed_ratio))
            modes.append((offset, float(speed_ratio)))

    plans, states = _expert_plans_batch_numpy(
        cond,
        wind_u,
        wind_v,
        np.asarray(waypoints, dtype=np.float32),
        np.asarray(speed_ratios, dtype=np.float32),
    )
    scores_array = expert_candidate_costs_batch_numpy(
        states, plans, cond,
        wind_u[:states.shape[-1]], wind_v[:states.shape[-1]],
    )
    order = np.argsort(scores_array)
    # Rank alternatives by route, not merely by a second speed on the same route.
    distinct_order = []
    used_offsets = set()
    for index in order:
        offset = round(float(modes[int(index)][0]), 3)
        if offset not in used_offsets:
            used_offsets.add(offset)
            distinct_order.append(int(index))
    selected = distinct_order[min(max(int(selection_rank), 0), len(distinct_order) - 1)]
    if stochastic and selection_rank == 0 and len(order) > 1:
        best = float(scores_array[order[0]])
        tolerance = config.EXPERT_RANDOM_TOLERANCE * max(abs(best), 0.1)
        eligible = [
            int(index) for index in order[:config.EXPERT_TOP_K]
            if float(scores_array[index]) <= best + tolerance
        ]
        if eligible:
            selected = int(np.random.choice(eligible))
    return plans[selected, :, :config.TRAJ_LEN], states[selected, :, :config.TRAJ_LEN], {
        "selected_index": np.asarray(selected, dtype=np.int64),
        "scores": scores_array,
        "modes": np.asarray(modes, dtype=np.float32),
        "controls": plans,
        "states": states,
        "order": order.astype(np.int64),
        "distinct_order": np.asarray(distinct_order, dtype=np.int64),
    }


def candidate_cost_torch(
    states: torch.Tensor,
    controls: torch.Tensor,
    cond: torch.Tensor,
    previous_plan: Optional[torch.Tensor] = None,
    wind_u: Optional[torch.Tensor] = None,
    wind_v: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Composite physical score used to select one of several DiT samples."""
    target = target_from_condition_torch(cond)
    start = cond[:, [config.COND_X, config.COND_Y]]
    end = states[:, :2, -1]
    start_dist = distance_km_torch(start, target)
    end_dist = distance_km_torch(end, target)
    domain = max(config.WIDTH_KM, config.HEIGHT_KM)
    progress = end_dist / domain

    x_km = states[:, 0] * config.WIDTH_KM
    y_km = states[:, 1] * config.HEIGHT_KM
    margin = config.BOUNDARY_MARGIN_KM
    edge_risk = (
        F.relu(margin - x_km)
        + F.relu(margin - (config.WIDTH_KM - x_km))
        + F.relu(margin - y_km)
        + F.relu(margin - (config.HEIGHT_KM - y_km))
    ) / margin
    # A single unsafe point matters because the first part of the plan is
    # executed before the next replan; averaging could hide it.
    boundary = edge_risk.max(dim=-1).values
    energy_risk = F.relu(config.ENERGY_SAFE_THRESHOLD - states[:, 5]).mean(dim=-1)
    terminal_hour = states[:, 4, -1] * 24.0
    required_reserve = required_energy_reserve_torch(terminal_hour, end_dist)
    reserve_deficit = F.relu(required_reserve - states[:, 5, -1])
    energy = energy_risk + reserve_deficit
    smooth = (controls[:, :, 1:] - controls[:, :, :-1]).square().mean(dim=(1, 2))
    if previous_plan is None:
        continuity = torch.zeros_like(progress)
    else:
        overlap = min(previous_plan.shape[-1], controls.shape[-1] - config.EXECUTE_STEPS)
        continuity = (controls[:, :, :overlap] - previous_plan[:, :, :overlap]).square().mean(dim=(1, 2))

    distances = distance_km_torch(states[:, :2].transpose(1, 2), target[:, None])
    distances = torch.cat([start_dist[:, None], distances], dim=1)
    regression = F.relu(distances[:, 1:] - distances[:, :-1]).mean(dim=-1)
    regression = regression / domain

    early_index = min(config.EXECUTE_STEPS, states.shape[-1]) - 1
    early_distance = distance_km_torch(states[:, :2, early_index], target)
    early_regression = F.relu(early_distance - start_dist) / domain
    useful_progress = F.relu(start_dist - end_dist)
    energy_spent = F.relu(cond[:, config.COND_ENERGY] - states[:, 5, -1])
    efficiency = (energy_spent / (useful_progress / 100.0).clamp_min(0.05)).clamp_max(5.0)
    trend_start = max(0, states.shape[-1] - 4)
    trend = F.relu(distances[:, -1] - distances[:, trend_start]) / domain
    stagnation = F.relu(config.STAGNATION_MIN_PROGRESS_KM - useful_progress) / domain
    closing = torch.zeros_like(progress)
    if wind_u is not None and wind_v is not None:
        final_index = min(states.shape[-1] - 1, wind_u.shape[1] - 1)
        terminal_pos = states[:, :2, -1]
        route_km = (target - terminal_pos) * torch.tensor(
            [config.WIDTH_KM, config.HEIGHT_KM],
            device=states.device, dtype=states.dtype,
        )
        direction = route_km / torch.linalg.vector_norm(
            route_km, dim=-1, keepdim=True
        ).clamp_min(1e-6)
        terminal_wind = torch.stack([
            _sample_wind_torch(wind_u[:, final_index], terminal_pos),
            _sample_wind_torch(wind_v[:, final_index], terminal_pos),
        ], dim=-1)
        terminal_ground = states[:, 2:4, -1] * config.MAX_SPEED + terminal_wind
        closing_speed = (terminal_ground * direction).sum(dim=-1)
        closing = F.relu(2.0 - closing_speed) / config.MAX_SPEED

    total = (
        config.SCORE_PROGRESS * progress
        + config.SCORE_BOUNDARY * boundary
        + config.SCORE_ENERGY * energy
        + config.SCORE_SMOOTH * smooth
        + config.SCORE_CONTINUITY * continuity
        + config.SCORE_REGRESSION * regression
        + config.SCORE_EARLY_PROGRESS * early_regression
        + config.SCORE_EFFICIENCY * efficiency
        + config.SCORE_TREND * trend
        + config.SCORE_CLOSING * closing
        + config.SCORE_STAGNATION * stagnation
    )
    return total, {
        "start_distance_km": start_dist,
        "end_distance_km": end_dist,
        "boundary": boundary,
        "energy": energy,
        "required_reserve": required_reserve,
        "efficiency": efficiency,
        "smooth": smooth,
        "continuity": continuity,
        "regression": regression,
        "early_regression": early_regression,
        "trend": trend,
        "closing": closing,
        "stagnation": stagnation,
    }


def trajectory_metrics(states: np.ndarray, target_pos) -> Dict[str, float]:
    """Metrics for a normalized state trajectory with shape (6,T)."""
    target_norm = np.array([target_pos[0] / config.WIDTH, target_pos[1] / config.HEIGHT])
    endpoint_error = distance_km_numpy(states[:2, -1], target_norm)
    dx = np.diff(states[0]) * config.WIDTH_KM
    dy = np.diff(states[1]) * config.HEIGHT_KM
    path_km = float(np.sqrt(dx * dx + dy * dy).sum())
    speed = np.linalg.norm(states[2:4], axis=0) * config.MAX_SPEED
    outside = (states[0] < 0) | (states[0] > 1) | (states[1] < 0) | (states[1] > 1)
    return {
        "endpoint_error_km": endpoint_error,
        "path_km": path_km,
        "mean_speed_mps": float(speed.mean()),
        "max_speed_mps": float(speed.max()),
        "min_energy": float(states[5].min()),
        "final_energy": float(states[5, -1]),
        "boundary_rate": float(outside.mean()),
    }
