"""Paired expert sweep for physically coupled airspeed and propulsion power.

This experiment deliberately runs only the rule expert.  It is intended to
choose a defensible physical envelope before regenerating demonstrations or
retraining DiT.  Every speed is evaluated on identical fixed-seed missions.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from batch_test import sample_task, wind_group
from dynamics.rolling import sample_wind_numpy
from dynamics.wind_field import WindGenerator
from rolling_planner import run_rule_planner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep maximum airspeed with a fixed cubic power coefficient"
    )
    parser.add_argument("--tests", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--speeds",
        default="15,18,20,25",
        help="Comma-separated maximum airspeeds in m/s",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="New result directory; defaults to a timestamped directory",
    )
    return parser.parse_args()


def parse_speeds(value: str):
    speeds = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not speeds or any(speed <= 0.0 for speed in speeds):
        raise ValueError("--speeds must contain positive comma-separated values")
    return speeds


def coupled_max_power(speed: float, reference_speed: float, reference_power: float):
    """Preserve P/v^3 so a given airspeed costs the same in every sweep arm."""
    return reference_power * (speed / reference_speed) ** 3


def frame_array(frames):
    return np.stack([
        frame.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if torch.is_tensor(frame)
        else np.asarray(frame).squeeze().astype(np.float32)
        for frame in frames
    ])


def field_wind_metrics(wind_u, wind_v, wind_start, required, maximum_speed):
    u = frame_array(wind_u[wind_start:wind_start + required])
    v = frame_array(wind_v[wind_start:wind_start + required])
    magnitude = np.sqrt(u * u + v * v)
    return {
        "field_wind_p95_mps": float(np.percentile(magnitude, 95.0)),
        "field_wind_p99_mps": float(np.percentile(magnitude, 99.0)),
        "field_wind_max_mps": float(np.max(magnitude)),
        "field_fraction_above_vmax": float(np.mean(magnitude > maximum_speed)),
    }


def encountered_wind_metrics(result, target_pos, maximum_speed):
    target_norm = np.asarray(target_pos, dtype=np.float32) / np.array(
        [config.WIDTH, config.HEIGHT], dtype=np.float32
    )
    magnitudes = []
    headwinds = []
    ground_closing_speeds = []
    # The last visualization frame repeats the terminal state, so omit it.
    for frame in result.frames[:-1]:
        position = np.asarray(frame["current"][:2], dtype=np.float32)
        wind = np.array([
            sample_wind_numpy(frame["wind_u"], position),
            sample_wind_numpy(frame["wind_v"], position),
        ], dtype=np.float32)
        direction = target_norm - position
        direction_norm = float(np.linalg.norm(direction))
        headwind = 0.0
        ground_closing = 0.0
        if direction_norm > 1e-8:
            unit_direction = direction / direction_norm
            headwind = max(0.0, -float(wind @ unit_direction))
            airspeed = np.asarray(frame["current"][2:4], dtype=np.float32) * maximum_speed
            ground_closing = float((airspeed + wind) @ unit_direction)
        magnitudes.append(float(np.linalg.norm(wind)))
        headwinds.append(headwind)
        ground_closing_speeds.append(ground_closing)
    if not magnitudes:
        return {
            "encountered_wind_mean_mps": 0.0,
            "encountered_wind_max_mps": 0.0,
            "encountered_headwind_max_mps": 0.0,
            "encountered_unopposable_fraction": 0.0,
            "negative_ground_closing_fraction": 0.0,
        }
    return {
        "encountered_wind_mean_mps": float(np.mean(magnitudes)),
        "encountered_wind_max_mps": float(np.max(magnitudes)),
        "encountered_headwind_max_mps": float(np.max(headwinds)),
        "encountered_unopposable_fraction": float(
            np.mean(np.asarray(headwinds) > maximum_speed)
        ),
        "negative_ground_closing_fraction": float(
            np.mean(np.asarray(ground_closing_speeds) < 0.0)
        ),
    }


def run_speed(speed, max_power, args, required):
    # Reset every RNG so all sweep arms receive exactly the same missions.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    generator = WindGenerator(
        config.WIDTH, config.HEIGHT,
        config.WIND_MIN_AVE_V, config.WIND_MAX_AVE_V,
        config.WIND_VARY_SPEED, config.WIND_VARY_NUM,
        10, 10, config.WIND_NUM_INP, config.RANGE_SCALE,
    )
    rows = []
    for test_id in range(args.tests):
        wind_u, wind_v, _, _ = generator.generate_wind_vector_list(config.WIND_HOURS)
        wind_start = rng.randint(0, len(wind_u) - required)
        start, goal, route_km = sample_task(rng)
        start_time = rng.uniform(0.0, 24.0)
        start_energy = rng.uniform(0.55, 1.0)
        result = run_rule_planner(
            wind_u, wind_v, start, goal,
            wind_start=wind_start,
            start_time=start_time,
            start_energy=start_energy,
        )
        group = wind_group(wind_u, wind_v, wind_start, start, goal)
        row = {
            "configured_max_speed_mps": speed,
            "configured_max_power_kw": max_power,
            "test": test_id,
            "start_x": start[0],
            "start_y": start[1],
            "goal_x": goal[0],
            "goal_y": goal[1],
            "wind_start": wind_start,
            "start_time": start_time,
            "start_energy": start_energy,
            "route_km": route_km,
            "route_group": (
                "short" if route_km < 150.0 else
                "medium" if route_km < 300.0 else "long"
            ),
            "wind_group": group,
            "termination": result.termination,
            **field_wind_metrics(
                wind_u, wind_v, wind_start, required, speed
            ),
            **encountered_wind_metrics(result, goal, speed),
            **result.metrics,
        }
        rows.append(row)
        print(
            f"vmax={speed:4.1f} test={test_id:02d} "
            f"{'OK' if result.success else result.termination.upper():8s} "
            f"wind={group:9s} time={row['mission_hours']:5.1f}h "
            f"error={row['endpoint_error_km']:7.1f}km "
            f"energy={row['final_energy'] * 100:5.1f}%"
        )
    return rows


def summarize(rows, speed, max_power):
    headwind = [row for row in rows if row["wind_group"] == "headwind"]
    successes = sum(row["success"] for row in rows)
    return {
        "configured_max_speed_mps": speed,
        "configured_max_power_kw": max_power,
        "tests": len(rows),
        "successes": int(successes),
        "success_rate": float(successes / len(rows)),
        "headwind_tests": len(headwind),
        "headwind_successes": int(sum(row["success"] for row in headwind)),
        "headwind_success_rate": float(
            sum(row["success"] for row in headwind) / len(headwind)
        ) if headwind else math.nan,
        "timeouts": sum(row["termination"] == "timeout" for row in rows),
        "energy_failures": sum(row["termination"] == "energy" for row in rows),
        "boundary_failures": sum(row["termination"] == "boundary" for row in rows),
        "mean_mission_hours": float(np.mean([row["mission_hours"] for row in rows])),
        "mean_endpoint_error_km": float(
            np.mean([row["endpoint_error_km"] for row in rows])
        ),
        "mean_final_energy": float(np.mean([row["final_energy"] for row in rows])),
        "mean_field_wind_p99_mps": float(
            np.mean([row["field_wind_p99_mps"] for row in rows])
        ),
        "max_field_wind_mps": float(
            np.max([row["field_wind_max_mps"] for row in rows])
        ),
        "mean_field_fraction_above_vmax": float(
            np.mean([row["field_fraction_above_vmax"] for row in rows])
        ),
        "max_encountered_headwind_mps": float(
            np.max([row["encountered_headwind_max_mps"] for row in rows])
        ),
        "mean_encountered_unopposable_fraction": float(
            np.mean([row["encountered_unopposable_fraction"] for row in rows])
        ),
        "mean_negative_ground_closing_fraction": float(
            np.mean([row["negative_ground_closing_fraction"] for row in rows])
        ),
    }


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    speeds = parse_speeds(args.speeds)
    if not args.output_dir:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(
            config.MODEL_SAVE_DIR, f"physical_parameter_sweep_{stamp}"
        )
    if os.path.exists(args.output_dir):
        raise FileExistsError(
            f"Refusing to overwrite existing sweep directory: {args.output_dir}"
        )
    os.makedirs(args.output_dir)

    reference_speed = float(config.MAX_SPEED)
    reference_power = float(config.MAX_POWER)
    coefficient = reference_power / reference_speed ** 3
    required = config.MAX_MISSION_STEPS + config.WIND_NUM_FRAMES + 1
    all_rows = []
    summaries = []
    try:
        for speed in speeds:
            max_power = coupled_max_power(speed, reference_speed, reference_power)
            config.MAX_SPEED = speed
            config.MAX_POWER = max_power
            rows = run_speed(speed, max_power, args, required)
            all_rows.extend(rows)
            summaries.append(summarize(rows, speed, max_power))
            write_csv(
                os.path.join(args.output_dir, f"expert_vmax_{speed:g}.csv"), rows
            )
    finally:
        config.MAX_SPEED = reference_speed
        config.MAX_POWER = reference_power

    write_csv(os.path.join(args.output_dir, "all_missions.csv"), all_rows)
    write_csv(os.path.join(args.output_dir, "summary.csv"), summaries)
    with open(
        os.path.join(args.output_dir, "experiment.json"),
        "w", encoding="utf-8",
    ) as handle:
        json.dump({
            "seed": args.seed,
            "tests_per_speed": args.tests,
            "speeds_mps": speeds,
            "reference_speed_mps": reference_speed,
            "reference_max_power_kw": reference_power,
            "propulsion_coefficient_kw_per_mps_cubed": coefficient,
            "power_rule": "Pmax(v)=reference_power*(v/reference_speed)^3",
            "max_mission_hours": config.MAX_MISSION_HOURS,
        }, handle, ensure_ascii=False, indent=2)

    print("\n=== Coupled physical-parameter sweep ===")
    for row in summaries:
        print(
            f"vmax={row['configured_max_speed_mps']:4.1f}m/s "
            f"pmax={row['configured_max_power_kw']:5.1f}kW "
            f"success={row['successes']:2d}/{row['tests']} "
            f"headwind={row['headwind_successes']:2d}/{row['headwind_tests']} "
            f"timeout={row['timeouts']:2d} energy={row['energy_failures']:2d} "
            f"mean_time={row['mean_mission_hours']:5.1f}h "
            f"mean_error={row['mean_endpoint_error_km']:6.1f}km"
        )
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
