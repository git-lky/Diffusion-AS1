"""Run one complete rolling-DiT mission and create static/animated results."""
import argparse
import math
import os
import random

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffusion_as1_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/diffusion_as1_cache")

import numpy as np
import torch

import config
from dynamics.wind_field import WindGenerator
from rolling_planner import load_diffusion, run_rolling_planner
from visualization import save_result_visualizations


def parse_args():
    parser = argparse.ArgumentParser(description="One-hour rolling DiT trajectory test")
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--steps", type=int, default=50, help="DDIM sampling steps")
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--candidates", type=int, default=config.INFERENCE_CANDIDATES)
    parser.add_argument("--physics-guidance", type=float, default=config.PHYSICS_GUIDANCE_SCALE)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--gif-fps", type=float, default=12.0)
    parser.add_argument("--output-dir", default=os.path.join(config.MODEL_SAVE_DIR, "rolling_result"))
    return parser.parse_args()


def random_task(rng):
    while True:
        start = [rng.randint(20, config.WIDTH - 20), rng.randint(20, config.HEIGHT - 20)]
        goal = [rng.randint(30, config.WIDTH - 30), rng.randint(30, config.HEIGHT - 30)]
        dx = (start[0] - goal[0]) / config.WIDTH * config.WIDTH_KM
        dy = (start[1] - goal[1]) / config.HEIGHT * config.HEIGHT_KM
        if math.hypot(dx, dy) >= 120.0:
            return start, goal


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)

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
    wind_u, wind_v, _, _ = wind_generator.generate_wind_vector_list(config.WIND_HOURS)
    required = config.MAX_MISSION_STEPS + config.WIND_NUM_FRAMES + 1
    wind_start = rng.randint(0, len(wind_u) - required)
    start, goal = random_task(rng)
    start_time = rng.uniform(0.0, 24.0)
    print(
        f"Start={start}, goal={goal}, start_time={start_time:.1f}h, "
        f"wind_frame={wind_start}, device={device}"
    )

    result = run_rolling_planner(
        diffusion, wind_u, wind_v, start, goal,
        wind_start=wind_start, start_time=start_time,
        guidance_scale=args.cfg, candidates=args.candidates,
        sampler=args.sampler, ddim_steps=args.steps,
        physics_guidance_scale=args.physics_guidance, device=device,
    )
    files = save_result_visualizations(result, args.output_dir, gif_fps=args.gif_fps)
    print(f"Termination: {result.termination}; success={result.success}")
    for name, value in result.metrics.items():
        print(f"{name}: {value:.4f}")
    for name, path in files.items():
        if path:
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
