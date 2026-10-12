"""PPO baseline evaluation on the same fixed scenarios as run_ablation.

Simulates the airship with the pretrained PPO policy (dynamics/ppo_policy.py)
for the full mission horizon and reports endpoint error / success / speed /
energy / path length, saved as CSV for comparison against the diffusion model.

Usage:
    python experiments/baseline_ppo.py --ppo_dir ./ppo_policy --scenarios 30 --seed 0
"""
import os, sys, math, random, argparse, csv
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from dynamics.airship import ASUnit_low
from dynamics.wind_field import WindGenerator
from dynamics.memory import Memory_wind_field
from dynamics.ppo_policy import PPOGuide
from data.dataset import TrajectoryDataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo_dir", default="./ppo_policy")
    ap.add_argument("--scenarios", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/results/baseline_ppo.csv")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    guide = PPOGuide(args.ppo_dir, device=device)
    ds = TrajectoryDataset(num_trajectories=1)
    airship = ASUnit_low(ds.mock_args)
    rng = random.Random(args.seed)

    wind_gen = WindGenerator(config.WIDTH, config.HEIGHT,
                             config.WIND_MIN_AVE_V, config.WIND_MAX_AVE_V,
                             config.WIND_VARY_SPEED, config.WIND_VARY_NUM,
                             10, 10, config.WIND_NUM_INP, config.RANGE_SCALE)
    wind_buffer = Memory_wind_field(10)
    for _ in range(10):
        wind_buffer.add_list(*wind_gen.generate_wind_vector_list(config.WIND_HOURS))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    print(f"{'#':>3s} {'Init':>12s} {'Target':>12s} {'Error':>8s} {'Speed':>6s} {'Energy':>7s} {'Result':>6s}")
    print("-" * 60)
    for i in range(args.scenarios):
        wx, wy, wvx, wvy = wind_buffer.get_list()
        max_start = len(wx) - config.MAX_MISSION_STEPS - config.WIND_NUM_FRAMES - 1
        wind_start = rng.randint(0, max(max_start, 1))
        while True:
            init_pos = [rng.randint(20, config.WIDTH - 20), rng.randint(20, config.HEIGHT - 20)]
            target_pos = [rng.randint(30, config.WIDTH - 30), rng.randint(30, config.HEIGHT - 30)]
            if math.sqrt((init_pos[0]-target_pos[0])**2 + (init_pos[1]-target_pos[1])**2) > 50:
                break

        airship.start_time = rng.uniform(0.0, 24.0)
        airship.reset(init_pos, target_pos)
        cur_as_state = np.array([
            airship.pos_state_normalize[0], airship.pos_state_normalize[1],
            airship.speed[0] / config.MAX_SPEED, airship.speed[1] / config.MAX_SPEED,
            airship.time / 24.0, airship.energy], dtype=np.float32)
        pos_log = [np.array(init_pos, dtype=float)]
        speed_log = []
        h_comp, w_comp = wx[0].shape[1], wx[0].shape[2]
        for step in range(config.MAX_MISSION_STEPS):
            wyi = min(h_comp - 1, max(0, int(airship.pos_state[1] / config.RANGE_SCALE)))
            wxi = min(w_comp - 1, max(0, int(airship.pos_state[0] / config.RANGE_SCALE)))
            wind_s = np.array([float(wx[wind_start+step][0, wyi, wxi]),
                               float(wy[wind_start+step][0, wyi, wxi])])
            wvx_i = wvx[wind_start+step] if wind_start+step < len(wvx) else np.zeros(252)
            wvy_i = wvy[wind_start+step] if wind_start+step < len(wvy) else np.zeros(252)
            action = guide.get_action(wvx_i, wvy_i, cur_as_state, target_pos, noise_scale=0.0)
            as_state, ct, ce, crs, cs, ws, sp = airship.step(action, wind_s, target_pos)
            cur_as_state = np.array([
                airship.pos_state_normalize[0], airship.pos_state_normalize[1],
                airship.speed[0] / config.MAX_SPEED, airship.speed[1] / config.MAX_SPEED,
                airship.time / 24.0, airship.energy], dtype=np.float32)
            pos_log.append(airship.pos_state.copy())
            speed_log.append(float(airship.real_speed))
            dx_km = (airship.pos_state[0] - target_pos[0]) / config.WIDTH * config.WIDTH_KM
            dy_km = (airship.pos_state[1] - target_pos[1]) / config.HEIGHT * config.HEIGHT_KM
            if math.hypot(dx_km, dy_km) <= config.TARGET_RANGE:
                break

        pos_log = np.array(pos_log)
        end = pos_log[-1]
        dx_km = (end[0] - target_pos[0]) / config.WIDTH * config.WIDTH_KM
        dy_km = (end[1] - target_pos[1]) / config.HEIGHT * config.HEIGHT_KM
        d_km = math.hypot(dx_km, dy_km)
        success = d_km <= config.TARGET_RANGE
        dx = np.diff(pos_log[:, 0]) / config.WIDTH * config.WIDTH_KM
        dy = np.diff(pos_log[:, 1]) / config.HEIGHT * config.HEIGHT_KM
        path_km = np.sum(np.sqrt(dx**2 + dy**2))
        avg_speed = float(np.mean(speed_log)) if speed_log else 0.0
        rows.append(dict(scenario=i, err_km=d_km, success=float(success),
                         speed=avg_speed, energy=float(airship.energy), path_km=path_km))
        status = "OK" if success else "FAIL"
        print(f"{i:3d} ({init_pos[0]:3d},{init_pos[1]:3d}) ({target_pos[0]:3d},{target_pos[1]:3d}) "
              f"{d_km:8.1f} {avg_speed:6.1f} {airship.energy*100:6.0f}% {status}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "err_km", "success", "speed", "energy", "path_km"])
        w.writeheader()
        w.writerows(rows)
    errs = [r["err_km"] for r in rows]
    print("\n=== PPO baseline summary ===")
    print(f"Success rate: {sum(r['success'] for r in rows)}/{len(rows)} = "
          f"{np.mean([r['success'] for r in rows])*100:.1f}%")
    print(f"Mean error: {np.mean(errs):.1f} km  Median: {np.median(errs):.1f} km")
    print(f"Mean path length: {np.mean([r['path_km'] for r in rows]):.0f} km")
    print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
