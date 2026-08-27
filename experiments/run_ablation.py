"""Ablation & comparison runner.

Evaluates a trained diffusion checkpoint on a fixed set of scenarios while
sweeping sampler type, DDIM steps, CFG scale and number of candidates.
Results are saved to CSV for comparison across model versions.

Usage:
    python experiments/run_ablation.py --ckpt checkpoints/model_final.pt \
        --samplers "ddim:20,ddim:50,ddpm:256" --cfgs "1.0,2.0,3.0" \
        --candidates "1,5,10" --scenarios 30 --seed 0
"""
import os, sys, math, random, argparse, csv, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models import create_denoiser
from models.wind_encoder import WindEncoder, TemporalWindEncoder
from models.diffusion import TrajectoryDiffusion
from dynamics.wind_field import WindGenerator
from dynamics.memory import Memory_wind_field
from batch_test import compute_metrics


def load_model(checkpoint_path, device):
    dit = create_denoiser().to(device)
    wind_encoder = (TemporalWindEncoder if config.WIND_NUM_FRAMES > 4 else WindEncoder)().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    dit.load_state_dict(ckpt["dit"])
    wind_encoder.load_state_dict(ckpt["wind_encoder"])
    dit.eval(); wind_encoder.eval()
    return TrajectoryDiffusion(dit, wind_encoder).to(device)


def make_scenarios(rng, wind_buffer, n):
    """Build n fixed scenarios: (cond, wind_u, wind_v, target_pos)."""
    scenarios = []
    for _ in range(n):
        wx, wy, wvx, wvy = wind_buffer.get_list()
        max_start = len(wx) - 1 - max(config.TRAJ_LEN - 1, max(config.WIND_FRAME_OFFSETS))
        wind_start = rng.randint(0, max(int(max_start), 1))
        while True:
            init_pos = [rng.randint(20, config.WIDTH - 20), rng.randint(20, config.HEIGHT - 20)]
            target_pos = [rng.randint(30, config.WIDTH - 30), rng.randint(30, config.HEIGHT - 30)]
            if math.sqrt((init_pos[0]-target_pos[0])**2 + (init_pos[1]-target_pos[1])**2) > 50:
                break
        cond_np = np.array([
            init_pos[0]/config.WIDTH, init_pos[1]/config.HEIGHT,
            0.0, 0.0, config.START_TIME/24.0, config.START_ENERGY,
            target_pos[0]/config.WIDTH, target_pos[1]/config.HEIGHT,
        ], dtype=np.float32)
        idx = [wind_start + o for o in config.WIND_FRAME_OFFSETS]
        wu = torch.stack([wx[i].float() for i in idx], dim=1).squeeze(2)
        wv = torch.stack([wy[i].float() for i in idx], dim=1).squeeze(2)
        scenarios.append((torch.tensor(cond_np).unsqueeze(0), wu, wv, target_pos))
    return scenarios


def evaluate(diffusion, device, cond, wu, wv, target_pos, cfg, num_samples, sampler, steps):
    cond = cond.to(device); wu = wu.to(device); wv = wv.to(device)
    t0 = time.time()
    with torch.no_grad():
        _, _, best = diffusion.sample_multi(cond, wu, wv, guidance_scale=cfg,
                                            num_samples=num_samples, sampler=sampler, ddim_steps=steps)
    traj_np = best[0].cpu().numpy()
    ex, ey = traj_np[0, -1] * config.WIDTH, traj_np[1, -1] * config.HEIGHT
    d_km = math.sqrt((ex-target_pos[0])**2 + (ey-target_pos[1])**2) / config.WIDTH * config.WIDTH_KM
    avg_speed, energy_end, path_len = compute_metrics(traj_np)
    return dict(err_km=d_km, success=float(d_km <= config.TARGET_RANGE),
                speed=avg_speed, energy=energy_end, path_km=path_len, time_s=time.time()-t0)


def parse_sweeps(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(config.MODEL_SAVE_DIR, "model_final.pt"))
    ap.add_argument("--scenarios", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samplers", default="ddim:50,ddpm:256", help="e.g. ddim:20,ddim:50,ddpm:256")
    ap.add_argument("--cfgs", default="1.0,2.0", help="CFG scales, e.g. 1.0,2.0,3.0")
    ap.add_argument("--candidates", default="1,5", help="candidate counts, e.g. 1,5,10")
    ap.add_argument("--out", default="experiments/results/ablation.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.ckpt):
        print(f"No checkpoint at {args.ckpt}")
        return
    diffusion = load_model(args.ckpt, device)
    print(f"Model loaded from {args.ckpt} ({device})")

    rng = random.Random(args.seed)
    wind_gen = WindGenerator(config.WIDTH, config.HEIGHT,
                             config.WIND_MIN_AVE_V, config.WIND_MAX_AVE_V,
                             config.WIND_VARY_SPEED, config.WIND_VARY_NUM,
                             10, 10, config.WIND_NUM_INP, config.RANGE_SCALE)
    wind_buffer = Memory_wind_field(10)
    for _ in range(10):
        wind_buffer.add_list(*wind_gen.generate_wind_vector_list(config.WIND_HOURS))

    scenarios = make_scenarios(rng, wind_buffer, args.scenarios)

    cfgs = [float(x) for x in parse_sweeps(args.cfgs)]
    cands = [int(x) for x in parse_sweeps(args.candidates)]
    samplers = []
    for s in parse_sweeps(args.samplers):
        name, _, steps = s.partition(":")
        samplers.append((s, name, int(steps) if steps else config.DIFFUSION_STEPS))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for si, (cond, wu, wv, target) in enumerate(scenarios):
        for sname, sampler, steps in samplers:
            for cfg in cfgs:
                for ncand in cands:
                    r = evaluate(diffusion, device, cond, wu, wv, target, cfg, ncand, sampler, steps)
                    rows.append(dict(scenario=si, config=f"{sname}|cfg={cfg}|k={ncand}", **r))
                    print(f"scen {si:2d} {sname:9s} cfg={cfg:3.1f} k={ncand:2d} -> "
                          f"err={r['err_km']:6.1f}km succ={r['success']} speed={r['speed']:4.1f} "
                          f"energy={r['energy']*100:3.0f}% path={r['path_km']:5.0f}km {r['time_s']:.1f}s")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "config", "err_km", "success", "speed", "energy", "path_km", "time_s"])
        w.writeheader()
        w.writerows(rows)

    print("\n=== Per-config summary ===")
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[r["config"]].append(r)
    print(f"{'config':26s} {'n':>3s} {'succ%':>6s} {'err_mean':>9s} {'err_med':>9s} {'speed':>6s} {'energy':>6s} {'path_km':>8s} {'time':>6s}")
    for name, rs in agg.items():
        print(f"{name:26s} {len(rs):3d} {np.mean([r['success'] for r in rs])*100:6.1f} "
              f"{np.mean([r['err_km'] for r in rs]):9.1f} {np.median([r['err_km'] for r in rs]):9.1f} "
              f"{np.mean([r['speed'] for r in rs]):6.1f} {np.mean([r['energy'] for r in rs])*100:6.0f} "
              f"{np.mean([r['path_km'] for r in rs]):8.0f} {np.mean([r['time_s'] for r in rs]):6.1f}")
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
