"""Batch evaluation of complete rolling-horizon missions."""
import argparse
import csv
import math
import os
import random
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffusion_as1_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/diffusion_as1_cache")

import numpy as np
import torch

import config
from dynamics.rolling import sample_wind_numpy
from dynamics.wind_field import WindGenerator
from rolling_planner import load_diffusion, run_rolling_planner, run_rule_planner
from visualization import save_animation, save_diagnostic_plot, save_static_result


def parse_args():
    parser = argparse.ArgumentParser(description="Batch rolling-DiT mission evaluation")
    parser.add_argument("--tests", type=int, default=30)
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--candidates", type=int, default=config.INFERENCE_CANDIDATES)
    parser.add_argument("--physics-guidance", type=float, default=config.PHYSICS_GUIDANCE_SCALE)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--animate-first", action="store_true")
    parser.add_argument(
        "--animate-failures", action="store_true",
        help="Create a GIF and diagnostic plot for every failed DiT mission",
    )
    parser.add_argument(
        "--animate-tests", default="",
        help="Comma-separated test IDs for which GIFs and diagnostic plots are created",
    )
    parser.add_argument(
        "--compare-expert", action="store_true",
        help="Evaluate the multi-route rule expert on the same scenarios",
    )
    parser.add_argument(
        "--animate-representative",
        action="store_true",
        help=(
            "Create GIFs and diagnostic plots for one median successful mission "
            "and one median failed mission"
        ),
    )
    parser.add_argument("--output-dir", default=os.path.join(config.MODEL_SAVE_DIR, "batch_rolling"))
    return parser.parse_args()


def sample_task(rng):
    while True:
        start = [rng.randint(20, config.WIDTH - 20), rng.randint(20, config.HEIGHT - 20)]
        goal = [rng.randint(30, config.WIDTH - 30), rng.randint(30, config.HEIGHT - 30)]
        dx = (start[0] - goal[0]) / config.WIDTH * config.WIDTH_KM
        dy = (start[1] - goal[1]) / config.HEIGHT * config.HEIGHT_KM
        distance = math.hypot(dx, dy)
        if distance >= 80.0:
            return start, goal, distance


def wind_group(wind_u_frames, wind_v_frames, wind_start, start, goal):
    """Classify average planning-window wind along the nominal route."""
    scale = np.array([config.WIDTH, config.HEIGHT], dtype=np.float32)
    start_norm = np.asarray(start, dtype=np.float32) / scale
    goal_norm = np.asarray(goal, dtype=np.float32) / scale
    route = (goal_norm - start_norm) * np.array(
        [config.WIDTH_KM, config.HEIGHT_KM], dtype=np.float32
    )
    winds = []
    for offset in range(config.TRAJ_LEN):
        fraction = (offset + 1) / config.TRAJ_LEN
        pos = start_norm + fraction * (goal_norm - start_norm)
        u = wind_u_frames[wind_start + offset].squeeze().numpy()
        v = wind_v_frames[wind_start + offset].squeeze().numpy()
        winds.append([sample_wind_numpy(u, pos), sample_wind_numpy(v, pos)])
    wind = np.mean(np.asarray(winds, dtype=np.float32), axis=0)
    cosine = float(wind @ route / (np.linalg.norm(wind) * np.linalg.norm(route) + 1e-6))
    return "tailwind" if cosine > 0.3 else "headwind" if cosine < -0.3 else "crosswind"


def grouped_report(rows, field):
    print(f"\n--- {field} ---")
    for group in sorted({row[field] for row in rows}):
        subset = [row for row in rows if row[field] == group]
        print(
            f"{group:16s} n={len(subset):3d} "
            f"success={np.mean([r['success'] for r in subset]) * 100:5.1f}% "
            f"error={np.mean([r['endpoint_error_km'] for r in subset]):7.1f}km "
            f"energy={np.mean([r['final_energy'] for r in subset]) * 100:5.1f}%"
        )


def representative_indices(rows):
    """Select non-extreme examples near the median endpoint error per outcome."""
    selected = []
    for success in (1, 0):
        subset = [row for row in rows if row["success"] == success]
        if not subset:
            continue
        if success:
            # Very short missions are not informative as the sole successful
            # animation. Prefer medium/long routes when available.
            informative = [row for row in subset if row["route_group"] != "short"]
            if informative:
                subset = informative
        median_error = float(np.median([row["endpoint_error_km"] for row in subset]))
        representative = min(
            subset,
            key=lambda row: (abs(row["endpoint_error_km"] - median_error), row["test"]),
        )
        selected.append(int(representative["test"]))
    return selected


def capture_rng_state(task_rng):
    """Keep enough small RNG state to reproduce one scenario without its frames."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "task": task_rng.getstate(),
    }


def restore_rng_state(state, task_rng):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    task_rng.setstate(state["task"])


def main():
    args = parse_args()
    try:
        animation_tests = {
            int(value.strip())
            for value in args.animate_tests.split(",")
            if value.strip()
        }
    except ValueError as error:
        raise ValueError("--animate-tests must contain comma-separated integers") from error
    invalid_animation_tests = sorted(
        test_id for test_id in animation_tests if not 0 <= test_id < args.tests
    )
    if invalid_animation_tests:
        raise ValueError(
            f"animation test IDs outside 0..{args.tests - 1}: "
            f"{invalid_animation_tests}"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint = os.path.join(config.MODEL_SAVE_DIR, "model_final.pt")
    if not os.path.exists(checkpoint):
        print(f"No rolling checkpoint at {checkpoint}. Run python train.py first.")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = load_diffusion(checkpoint, device)

    wind_generator = WindGenerator(
        config.WIDTH, config.HEIGHT,
        config.WIND_MIN_AVE_V, config.WIND_MAX_AVE_V,
        config.WIND_VARY_SPEED, config.WIND_VARY_NUM,
        10, 10, config.WIND_NUM_INP, config.RANGE_SCALE,
    )
    rows = []
    expert_rows = []
    scenario_states = [] if args.animate_representative else None
    required = config.MAX_MISSION_STEPS + config.WIND_NUM_FRAMES + 1
    print(
        f"Testing {args.tests} rolling missions | {config.PLAN_HOURS}h prediction / "
        f"{config.REPLAN_MINUTES}min execution | {args.sampler} | candidates={args.candidates}"
    )

    for test_id in range(args.tests):
        if scenario_states is not None:
            scenario_states.append(capture_rng_state(rng))
        wind_u, wind_v, _, _ = wind_generator.generate_wind_vector_list(config.WIND_HOURS)
        wind_start = rng.randint(0, len(wind_u) - required)
        start, goal, route_km = sample_task(rng)
        start_time = rng.uniform(0.0, 24.0)
        start_energy = rng.uniform(0.55, 1.0)
        begin = time.time()
        result = run_rolling_planner(
            diffusion, wind_u, wind_v, start, goal,
            wind_start=wind_start, start_time=start_time, start_energy=start_energy,
            guidance_scale=args.cfg, candidates=args.candidates,
            sampler=args.sampler, ddim_steps=args.steps,
            physics_guidance_scale=args.physics_guidance, device=device,
        )
        elapsed = time.time() - begin
        route_group = "short" if route_km < 150 else "medium" if route_km < 300 else "long"
        row = {
            "test": test_id,
            "success": int(result.success),
            "termination": result.termination,
            "route_group": route_group,
            "wind_group": wind_group(wind_u, wind_v, wind_start, start, goal),
            "route_km": route_km,
            "start_time": start_time,
            "start_energy": start_energy,
            "runtime_s": elapsed,
            **result.metrics,
        }
        rows.append(row)
        if args.compare_expert:
            expert_begin = time.time()
            expert_result = run_rule_planner(
                wind_u, wind_v, start, goal,
                wind_start=wind_start, start_time=start_time,
                start_energy=start_energy,
            )
            expert_rows.append({
                "test": test_id,
                "success": int(expert_result.success),
                "termination": expert_result.termination,
                "route_group": route_group,
                "wind_group": row["wind_group"],
                "route_km": route_km,
                "start_time": start_time,
                "start_energy": start_energy,
                "runtime_s": time.time() - expert_begin,
                **expert_result.metrics,
            })
        status = "OK" if result.success else result.termination.upper()
        stem = f"test_{test_id:03d}_{status}"
        save_static_result(result, os.path.join(args.output_dir, stem + ".png"))
        should_animate = (
            (args.animate_first and test_id == 0)
            or (args.animate_failures and not result.success)
            or test_id in animation_tests
        )
        if should_animate:
            save_animation(
                result,
                os.path.join(args.output_dir, stem + ".gif"),
            )
            save_diagnostic_plot(result, os.path.join(args.output_dir, stem + "_diagnostics.png"))
        print(
            f"{test_id:3d} {status:8s} error={row['endpoint_error_km']:7.1f}km "
            f"energy={row['final_energy'] * 100:5.1f}% path={row['path_km']:7.1f}km "
            f"replans={int(row['replans']):2d} runtime={elapsed:.1f}s"
        )

    if args.animate_representative:
        selected = representative_indices(rows)
        print(f"\nCreating representative animations for tests: {selected}")
        for test_id in selected:
            restore_rng_state(scenario_states[test_id], rng)
            wind_u, wind_v, _, _ = wind_generator.generate_wind_vector_list(config.WIND_HOURS)
            wind_start = rng.randint(0, len(wind_u) - required)
            start, goal, _ = sample_task(rng)
            start_time = rng.uniform(0.0, 24.0)
            start_energy = rng.uniform(0.55, 1.0)
            result = run_rolling_planner(
                diffusion, wind_u, wind_v, start, goal,
                wind_start=wind_start, start_time=start_time, start_energy=start_energy,
                guidance_scale=args.cfg, candidates=args.candidates,
                sampler=args.sampler, ddim_steps=args.steps,
                physics_guidance_scale=args.physics_guidance, device=device,
            )
            row = rows[test_id]
            status = "OK" if row["success"] else row["termination"].upper()
            stem = f"test_{test_id:03d}_{status}"
            save_animation(result, os.path.join(args.output_dir, stem + ".gif"))
            save_diagnostic_plot(
                result,
                os.path.join(args.output_dir, stem + "_diagnostics.png"),
            )

    csv_path = os.path.join(args.output_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if expert_rows:
        expert_csv = os.path.join(args.output_dir, "expert_metrics.csv")
        with open(expert_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(expert_rows[0].keys()))
            writer.writeheader()
            writer.writerows(expert_rows)

    print("\n=== Summary ===")
    print(f"Success: {sum(r['success'] for r in rows)}/{len(rows)} = {np.mean([r['success'] for r in rows]) * 100:.1f}%")
    print(f"Mean endpoint error: {np.mean([r['endpoint_error_km'] for r in rows]):.1f} km")
    print(f"Mean final energy: {np.mean([r['final_energy'] for r in rows]) * 100:.1f}%")
    print(f"Mean runtime: {np.mean([r['runtime_s'] for r in rows]):.1f} s/mission")
    grouped_report(rows, "route_group")
    grouped_report(rows, "wind_group")
    if expert_rows:
        print("\n=== Multi-route expert baseline ===")
        print(
            f"Success: {sum(r['success'] for r in expert_rows)}/{len(expert_rows)} = "
            f"{np.mean([r['success'] for r in expert_rows]) * 100:.1f}%"
        )
        print(
            f"Mean endpoint error: "
            f"{np.mean([r['endpoint_error_km'] for r in expert_rows]):.1f} km"
        )
        grouped_report(expert_rows, "wind_group")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
