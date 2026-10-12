"""Fixed-scenario ablation study for the rolling DiT planner."""
import argparse
import csv
import math
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from dynamics.wind_field import WindGenerator
from rolling_planner import load_diffusion, run_rolling_planner


def comma_values(text, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def make_scenarios(rng, count, wind_length):
    required = config.MAX_MISSION_STEPS + config.WIND_NUM_FRAMES + 1
    scenarios = []
    for _ in range(count):
        while True:
            start = [rng.randint(20, config.WIDTH - 20), rng.randint(20, config.HEIGHT - 20)]
            goal = [rng.randint(30, config.WIDTH - 30), rng.randint(30, config.HEIGHT - 30)]
            dx = (start[0] - goal[0]) / config.WIDTH * config.WIDTH_KM
            dy = (start[1] - goal[1]) / config.HEIGHT * config.HEIGHT_KM
            if math.hypot(dx, dy) >= 100:
                break
        scenarios.append({
            "start": start,
            "goal": goal,
            "wind_start": rng.randint(0, wind_length - required),
            "start_time": rng.uniform(0.0, 24.0),
        })
    return scenarios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=os.path.join(config.MODEL_SAVE_DIR, "model_final.pt"))
    parser.add_argument("--scenarios", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samplers", default="ddim:20,ddim:50")
    parser.add_argument("--cfgs", default="1.0,2.0")
    parser.add_argument("--candidates", default="1,5")
    parser.add_argument("--physics-guidance", default="0.0")
    parser.add_argument("--out", default="experiments/results/rolling_ablation.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = load_diffusion(args.ckpt, device)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    generator = WindGenerator(
        config.WIDTH, config.HEIGHT,
        config.WIND_MIN_AVE_V, config.WIND_MAX_AVE_V,
        config.WIND_VARY_SPEED, config.WIND_VARY_NUM,
        10, 10, config.WIND_NUM_INP, config.RANGE_SCALE,
    )
    wind_u, wind_v, _, _ = generator.generate_wind_vector_list(config.WIND_HOURS)
    scenarios = make_scenarios(rng, args.scenarios, len(wind_u))
    samplers = []
    for value in comma_values(args.samplers, str):
        name, _, steps = value.partition(":")
        samplers.append((name, int(steps or config.DIFFUSION_STEPS)))

    rows = []
    for scenario_id, scenario in enumerate(scenarios):
        for sampler, steps in samplers:
            for cfg_scale in comma_values(args.cfgs, float):
                for candidates in comma_values(args.candidates, int):
                    for physics_scale in comma_values(args.physics_guidance, float):
                        begin = time.time()
                        result = run_rolling_planner(
                            diffusion, wind_u, wind_v,
                            scenario["start"], scenario["goal"],
                            wind_start=scenario["wind_start"],
                            start_time=scenario["start_time"],
                            guidance_scale=cfg_scale,
                            candidates=candidates,
                            sampler=sampler,
                            ddim_steps=steps,
                            physics_guidance_scale=physics_scale,
                            device=device,
                        )
                        row = {
                            "scenario": scenario_id, "sampler": sampler, "steps": steps,
                            "cfg": cfg_scale, "candidates": candidates,
                            "physics_guidance": physics_scale,
                            "runtime_s": time.time() - begin,
                            "termination": result.termination,
                            **result.metrics,
                        }
                        rows.append(row)
                        print(
                            f"scenario={scenario_id} {sampler}:{steps} cfg={cfg_scale} "
                            f"k={candidates} pg={physics_scale} success={result.success} "
                            f"error={result.metrics['endpoint_error_km']:.1f}km"
                        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
