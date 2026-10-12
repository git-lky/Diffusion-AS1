"""Short train-to-inference gate that must pass before a full rolling-DiT run."""
from __future__ import annotations

import argparse
import itertools
import math
import os
import random

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffusion_as1_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/diffusion_as1_cache")

import numpy as np
import torch

import config
from data.dataset import RollingWindowDataset, TrajectoryDataset
from dynamics.rolling import distance_km_torch, target_from_condition_torch
from dynamics.wind_field import WindGenerator
from models import create_denoiser
from models.diffusion import TrajectoryDiffusion
from models.wind_encoder import SpatioTemporalWindEncoder
from rolling_planner import run_rolling_planner, run_rule_planner


def parse_args():
    parser = argparse.ArgumentParser(description="Rolling-DiT preflight training gate")
    parser.add_argument("--missions", type=int, default=96)
    parser.add_argument(
        "--updates", type=int, default=1200,
        help="short-run optimizer updates; about 60-70 epochs on the default smoke dataset",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rolling-tests", type=int, default=3)
    parser.add_argument(
        "--cache-dir", default="/tmp/diffusion_as1_rolling_v7_preflight",
        help="temporary small-dataset directory; never uses the formal training cache",
    )
    return parser.parse_args()


def seed_everything():
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)


def move(batch, device):
    controls, cond, wind_u, wind_v, states, valid, previous, previous_mask = batch
    return (
        controls.to(device).permute(0, 2, 1),
        cond.to(device), wind_u.to(device, dtype=torch.float32),
        wind_v.to(device, dtype=torch.float32),
        states.to(device).permute(0, 2, 1), valid.to(device),
        previous.to(device).permute(0, 2, 1), previous_mask.to(device),
    )


@torch.no_grad()
def generation_metrics(diffusion, batch, device):
    _, cond, wind_u, wind_v, _, _, _, _ = move(batch, device)
    previous = torch.zeros(
        cond.shape[0], config.ACTION_DIM, config.TRAJ_LEN,
        device=device, dtype=cond.dtype,
    )
    sampled = diffusion.sample_multi(
        cond, wind_u, wind_v, previous,
        guidance_scale=1.5, num_samples=3,
        sampler="ddim", ddim_steps=25, physics_guidance_scale=0.0,
    )
    states = sampled["best_states"]
    target = target_from_condition_torch(cond)
    progress = (
        distance_km_torch(cond[:, :2], target)
        - distance_km_torch(states[:, :2, -1], target)
    )
    outside = (
        (states[:, 0] < 0.0) | (states[:, 0] > 1.0)
        | (states[:, 1] < 0.0) | (states[:, 1] > 1.0)
    ).any(dim=1)
    return {
        "mean_progress_km": float(progress.mean()),
        "positive_fraction": float((progress > 0.0).float().mean()),
        "boundary_fraction": float(outside.float().mean()),
        "feasible_fraction": float(sampled["feasible"].any(dim=1).float().mean()),
    }


def main():
    args = parse_args()
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Preflight device: {device}")

    source = TrajectoryDataset(num_trajectories=args.missions, cache_dir=args.cache_dir)
    arrays = source.load() if os.path.exists(source.cache_file) else source.generate_all()
    source.quality_report(arrays, strict=True)
    cold_validation = np.flatnonzero(
        (arrays["split"] == 1) & (~arrays["previous_mask"].any(axis=1))
    )
    if len(cold_validation) < 4:
        raise RuntimeError("preflight needs at least four cold-start validation windows")
    train_indices = np.flatnonzero(arrays["split"] == 0)
    train_loader = torch.utils.data.DataLoader(
        RollingWindowDataset(arrays, train_indices),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    validation_loader = torch.utils.data.DataLoader(
        RollingWindowDataset(arrays, cold_validation[:16]),
        batch_size=min(16, len(cold_validation)), shuffle=False, num_workers=0,
    )

    diffusion = TrajectoryDiffusion(
        create_denoiser(), SpatioTemporalWindEncoder()
    ).to(device)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=config.LEARNING_RATE)
    losses = []
    batches = itertools.cycle(train_loader)
    for update in range(args.updates):
        controls, cond, wind_u, wind_v, states, valid, previous, previous_mask = move(
            next(batches), device
        )
        result = diffusion.loss(
            controls, cond, wind_u, wind_v, states, valid, previous, previous_mask
        )
        optimizer.zero_grad(set_to_none=True)
        result["total"].backward()
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), config.GRAD_CLIP)
        optimizer.step()
        losses.append(float(result["total"]))
        if (update + 1) % 50 == 0:
            print(f"update={update + 1:3d} mean_loss={np.mean(losses[-25:]):.6f}")

    initial_loss = float(np.mean(losses[:25]))
    final_loss = float(np.mean(losses[-25:]))
    diffusion.eval()
    metrics = generation_metrics(diffusion, next(iter(validation_loader)), device)
    print(f"Loss: {initial_loss:.6f} -> {final_loss:.6f}")
    print(
        f"Generation: progress={metrics['mean_progress_km']:.1f}km "
        f"positive={metrics['positive_fraction']:.1%} "
        f"safe={1.0 - metrics['boundary_fraction']:.1%} "
        f"feasible={metrics['feasible_fraction']:.1%}"
    )
    failures = []
    if final_loss >= initial_loss * 0.65:
        failures.append("loss did not fall by at least 35%")
    if metrics["mean_progress_km"] <= 5.0:
        failures.append("free-running samples do not make useful progress")
    if metrics["positive_fraction"] < 0.70:
        failures.append("fewer than 70% of cold-start samples move toward the goal")
    if metrics["boundary_fraction"] > 0.10:
        failures.append("more than 10% of samples cross the boundary")
    if metrics["feasible_fraction"] < 0.90:
        failures.append("candidate feasibility is below 90%")

    rolling_seed = config.RANDOM_SEED + 97
    random.seed(rolling_seed)
    np.random.seed(rolling_seed)
    torch.manual_seed(rolling_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rolling_seed)
    wind_generator = WindGenerator(
        config.WIDTH, config.HEIGHT,
        config.WIND_MIN_AVE_V, config.WIND_MAX_AVE_V,
        config.WIND_VARY_SPEED, config.WIND_VARY_NUM,
        10, 10, config.WIND_NUM_INP, config.RANGE_SCALE,
    )
    task_rng = random.Random(rolling_seed)
    rolling_results = []
    expert_results = []
    required = config.MAX_MISSION_STEPS + config.WIND_NUM_FRAMES + 1
    for test_id in range(args.rolling_tests):
        wind_u, wind_v, _, _ = wind_generator.generate_wind_vector_list(config.WIND_HOURS)
        wind_start = task_rng.randint(0, len(wind_u) - required)
        while True:
            start = [
                task_rng.randint(20, config.WIDTH - 20),
                task_rng.randint(20, config.HEIGHT - 20),
            ]
            goal = [
                task_rng.randint(30, config.WIDTH - 30),
                task_rng.randint(30, config.HEIGHT - 30),
            ]
            dx = (start[0] - goal[0]) / config.WIDTH * config.WIDTH_KM
            dy = (start[1] - goal[1]) / config.HEIGHT * config.HEIGHT_KM
            if math.hypot(dx, dy) >= 80.0:
                break
        start_time = task_rng.uniform(0.0, 24.0)
        start_energy = task_rng.uniform(0.55, 1.0)
        result = run_rolling_planner(
            diffusion, wind_u, wind_v, start, goal,
            wind_start=wind_start, start_time=start_time,
            start_energy=start_energy,
            guidance_scale=1.5, candidates=3, sampler="ddim", ddim_steps=25,
            physics_guidance_scale=0.0, device=device,
        )
        expert = run_rule_planner(
            wind_u, wind_v, start, goal,
            wind_start=wind_start, start_time=start_time,
            start_energy=start_energy,
        )
        rolling_results.append(result)
        expert_results.append(expert)
        print(
            f"rolling={test_id} termination={result.termination} "
            f"error={result.metrics['endpoint_error_km']:.1f}km "
            f"resets={int(result.metrics['plan_reset_count'])} "
            f"infeasible={int(result.metrics['infeasible_plan_count'])} | "
            f"expert={expert.termination} "
            f"expert_error={expert.metrics['endpoint_error_km']:.1f}km"
        )
    rolling_successes = sum(result.success for result in rolling_results)
    rolling_boundaries = sum(result.termination == "boundary" for result in rolling_results)
    expert_successes = sum(result.success for result in expert_results)
    required_successes = math.ceil(expert_successes * 0.5)
    if expert_successes < math.ceil(args.rolling_tests * 0.5):
        failures.append(
            f"expert reached only {expert_successes}/{args.rolling_tests} rolling targets"
        )
    if rolling_successes < required_successes:
        failures.append(
            f"only {rolling_successes}/{args.rolling_tests} complete rolling tasks reached the goal"
        )
    if rolling_boundaries > 0:
        failures.append(f"{rolling_boundaries} complete rolling tasks crossed the boundary")
    if failures:
        raise RuntimeError("preflight failed: " + "; ".join(failures))
    print("PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
