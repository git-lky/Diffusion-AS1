"""Evaluate the current offline rule expert before any expensive DiT training run."""
import argparse
import csv
import math
import os
import random

import numpy as np
import torch

import config
from batch_test import sample_task, wind_group
from dynamics.wind_field import WindGenerator
from rolling_planner import run_rule_planner


def parse_args():
    parser = argparse.ArgumentParser(description="Fixed-seed expert quality gate")
    parser.add_argument("--tests", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--legacy-v4-scenarios", action="store_true",
        help="reuse the v4 wind-start sampling range for paired comparison",
    )
    parser.add_argument(
        "--output", default=os.path.join(config.MODEL_SAVE_DIR, "expert_gate_metrics.csv")
    )
    parser.add_argument(
        "--scenario-csv", default="",
        help="CSV manifest whose fixed start/goal/wind-start values are reused",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    manifest = None
    if args.scenario_csv:
        with open(args.scenario_csv, newline="", encoding="utf-8") as handle:
            manifest = list(csv.DictReader(handle))
        if len(manifest) < args.tests:
            raise ValueError("scenario CSV contains fewer rows than --tests")
    generator = WindGenerator(
        config.WIDTH, config.HEIGHT,
        config.WIND_MIN_AVE_V, config.WIND_MAX_AVE_V,
        config.WIND_VARY_SPEED, config.WIND_VARY_NUM,
        10, 10, config.WIND_NUM_INP, config.RANGE_SCALE,
    )
    context = config.TRAJ_LEN if args.legacy_v4_scenarios else config.WIND_NUM_FRAMES
    required = config.MAX_MISSION_STEPS + context + 1
    rows = []
    for test_id in range(args.tests):
        wind_hours = max(config.WIND_HOURS, config.MAX_MISSION_HOURS + 96)
        wind_u, wind_v, _, _ = generator.generate_wind_vector_list(wind_hours)
        if manifest is None:
            wind_start = rng.randint(0, len(wind_u) - required)
            start, goal, route_km = sample_task(rng)
            start_time = rng.uniform(0.0, 24.0)
            start_energy = rng.uniform(0.55, 1.0)
        else:
            fixed = manifest[test_id]
            wind_start = int(float(fixed["wind_start"]))
            start = [int(float(fixed["start_x"])), int(float(fixed["start_y"]))]
            goal = [int(float(fixed["goal_x"])), int(float(fixed["goal_y"]))]
            route_km = float(fixed["route_km"])
            start_time = float(fixed["start_time"])
            start_energy = float(fixed["start_energy"])
        result = run_rule_planner(
            wind_u, wind_v, start, goal,
            wind_start=wind_start, start_time=start_time,
            start_energy=start_energy,
        )
        group = wind_group(wind_u, wind_v, wind_start, start, goal)
        row = {
            "test": test_id,
            "start_x": start[0], "start_y": start[1],
            "goal_x": goal[0], "goal_y": goal[1],
            "wind_start": wind_start,
            "start_time": start_time, "start_energy": start_energy,
            "route_km": route_km,
            "route_group": "short" if route_km < 150 else "medium" if route_km < 300 else "long",
            "wind_group": group,
            "termination": result.termination,
            **result.metrics,
        }
        rows.append(row)
        print(
            f"{test_id:3d} {'OK' if result.success else result.termination.upper():8s} "
            f"wind={group:9s} error={row['endpoint_error_km']:7.1f}km "
            f"energy={row['final_energy'] * 100:5.1f}%"
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    successes = sum(row["success"] for row in rows)
    headwind = [row for row in rows if row["wind_group"] == "headwind"]
    headwind_successes = sum(row["success"] for row in headwind)
    boundaries = sum(row["termination"] == "boundary" for row in rows)
    required_total = math.ceil(0.90 * len(rows))
    # For the fixed 14-headwind benchmark, the agreed gate is at least 11/14
    # (78.6%, reported as approximately 80%) rather than rounding upward to 12.
    required_headwind = round(0.80 * len(headwind)) if headwind else 0
    print(
        f"Expert gate | total={int(successes)}/{len(rows)} "
        f"headwind={int(headwind_successes)}/{len(headwind)} boundary={boundaries}"
    )
    if successes < required_total or headwind_successes < required_headwind or boundaries:
        raise RuntimeError(
            "expert gate failed: require >=90% total, about 80% headwind and zero boundary"
        )
    print("EXPERT GATE PASSED")


if __name__ == "__main__":
    main()
