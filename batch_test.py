"""Batch test with multi-trajectory visualization, wind field overlay and grouped metrics."""
import os, sys, math, random, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models import create_denoiser
from models.wind_encoder import WindEncoder, TemporalWindEncoder
from models.diffusion import TrajectoryDiffusion
from dynamics.wind_field import WindGenerator
from dynamics.memory import Memory_wind_field

NUM_CANDIDATES = 5
WIND_FRAME_OFFSETS = config.WIND_FRAME_OFFSETS

def parse_args():
    p = argparse.ArgumentParser(description="Batch test with grouping and multi-metrics")
    p.add_argument("--tests", type=int, default=30, help="number of random scenarios")
    p.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm",
                   help="denoising sampler: full DDPM (256 steps) or accelerated DDIM")
    p.add_argument("--steps", type=int, default=50, help="DDIM steps (used when --sampler ddim)")
    p.add_argument("--cfg", type=float, default=2.0, help="CFG guidance scale")
    p.add_argument("--candidates", type=int, default=5, help="candidate trajectories per scenario")
    p.add_argument("--seed", type=int, default=None, help="random seed for reproducible scenarios")
    p.add_argument("--output-dir", type=str, default=None,
                   help="image output directory (default: checkpoints/batch_results)")
    return p.parse_args()

def group_report(title, tags, group_names, all_dist, all_success, all_avg_speed, all_energy, all_path_len):
    """Print per-group success rate and metric means. tags align with per-test results."""
    print(f"\n--- {title} ---")
    print(f"{'group':10s} {'n':>3s} {'succ%':>6s} {'err_mean':>9s} {'err_med':>9s} "
          f"{'speed':>6s} {'energy':>6s} {'path_km':>8s}")
    for g in group_names:
        idx = [i for i, t in enumerate(tags) if t == g]
        if not idx:
            print(f"{g:10s} {0:3d}")
            continue
        d = [all_dist[i] for i in idx]
        s = [all_success[i] for i in idx]
        sp = [all_avg_speed[i] for i in idx]
        e = [all_energy[i] for i in idx]
        pl = [all_path_len[i] for i in idx]
        print(f"{g:10s} {len(idx):3d} {np.mean(s)*100:6.1f} {np.mean(d):9.1f} {np.median(d):9.1f} "
              f"{np.mean(sp):6.1f} {np.mean(e)*100:6.0f} {np.mean(pl):8.0f}")

def load_model(checkpoint_path, device):
    dit = create_denoiser().to(device)
    wind_encoder = (TemporalWindEncoder if config.WIND_NUM_FRAMES > 4 else WindEncoder)().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    dit.load_state_dict(ckpt["dit"])
    wind_encoder.load_state_dict(ckpt["wind_encoder"])
    dit.eval(); wind_encoder.eval()
    return TrajectoryDiffusion(dit, wind_encoder).to(device)

def plot_multi_trajectory(all_trajs_np, best_traj_np, init_pos, target_pos,
                           wind_u_frame, wind_v_frame, d_km, success, save_path, test_id):
    """Plot all candidate trajectories (thin translucent) + best (thick) + wind quiver."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Wind field quiver (every other grid point for clarity)
    step = 2
    H, W = wind_u_frame.shape
    Y, X = np.meshgrid(np.arange(0, H, step), np.arange(0, W, step), indexing='ij')
    # Map wind grid coords to pixel coords
    X_pix = (X + 0.5) * config.RANGE_SCALE
    Y_pix = (Y + 0.5) * config.RANGE_SCALE
    U = wind_u_frame[::step, ::step]
    V = wind_v_frame[::step, ::step]
    ax.quiver(X_pix, Y_pix, U, V, scale=100, width=0.003,
              color='lightsteelblue', alpha=0.7, zorder=1)

    # All candidate trajectories (thin translucent)
    for k in range(all_trajs_np.shape[0]):
        xs = all_trajs_np[k, 0] * config.WIDTH
        ys = all_trajs_np[k, 1] * config.HEIGHT
        ax.plot(xs, ys, '-', linewidth=0.6, alpha=0.35, color='steelblue', zorder=2)
        # Endpoint dot
        ax.plot(xs[-1], ys[-1], 'o', markersize=2.5, alpha=0.5, color='steelblue', zorder=2)

    # Best trajectory (thick)
    bxs = best_traj_np[0] * config.WIDTH
    bys = best_traj_np[1] * config.HEIGHT
    ax.plot(bxs, bys, '-', linewidth=2.0, color='darkblue', zorder=3, label='Best')
    ax.plot(bxs[-1], bys[-1], 's', color='darkorange', markersize=7, zorder=4, label='Endpoint')

    # Start / target
    ax.plot(init_pos[0], init_pos[1], 'go', markersize=10, zorder=5, label='Start')
    ax.plot(target_pos[0], target_pos[1], 'r*', markersize=14, zorder=5, label='Target')

    # Arrival circle
    r_pix = config.TARGET_RANGE / config.WIDTH_KM * config.WIDTH
    circle = plt.Circle((target_pos[0], target_pos[1]), r_pix,
                        color='r', fill=False, linestyle='--', linewidth=0.8, alpha=0.5, zorder=4)
    ax.add_patch(circle)

    status = "OK" if success else "FAIL"
    ax.set_title(f"Test #{test_id}: {status} | Error: {d_km:.1f} km")
    ax.set_xlim(0, config.WIDTH); ax.set_ylim(0, config.HEIGHT)
    ax.set_xlabel("X (pixel)"); ax.set_ylabel("Y (pixel)")
    ax.legend(loc='upper right', fontsize=7); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def compute_metrics(traj_np):
    """Compute extra metrics from a generated trajectory."""
    xs = traj_np[0] * config.WIDTH
    ys = traj_np[1] * config.HEIGHT
    speeds = np.sqrt(traj_np[2]**2 + traj_np[3]**2) * config.MAX_SPEED
    avg_speed = float(np.mean(speeds))
    energy_end = float(traj_np[5, -1])
    # Total distance along trajectory
    dx = np.diff(xs); dy = np.diff(ys)
    total_dist_pix = float(np.sum(np.sqrt(dx**2 + dy**2)))
    total_dist_km = total_dist_pix / config.WIDTH * config.WIDTH_KM
    return avg_speed, energy_end, total_dist_km


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(config.MODEL_SAVE_DIR, "model_final.pt")
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint at {ckpt_path}")
        return
    diffusion = load_model(ckpt_path, device)
    print(f"Model loaded ({device})")

    img_dir = args.output_dir or os.path.join(config.MODEL_SAVE_DIR, "batch_results")
    os.makedirs(img_dir, exist_ok=True)

    # Generate wind fields
    wind_gen = WindGenerator(config.WIDTH, config.HEIGHT,
        config.WIND_MIN_AVE_V, config.WIND_MAX_AVE_V,
        config.WIND_VARY_SPEED, config.WIND_VARY_NUM,
        10, 10, config.WIND_NUM_INP, config.RANGE_SCALE)
    wind_buffer = Memory_wind_field(10)
    for _ in range(10):
        wx, wy, wvx, wvy = wind_gen.generate_wind_vector_list(config.WIND_HOURS)
        wind_buffer.add_list(wx, wy, wvx, wvy)

    NUM_TESTS = args.tests
    SAMPLER = args.sampler
    DDIM_STEPS = args.steps
    CFG_SCALE = args.cfg
    NUM_CANDIDATES = args.candidates
    distances_km = []
    success_count = 0
    all_init = []; all_target = []; all_dist = []; all_success = []
    all_avg_speed = []; all_energy = []; all_path_len = []
    all_dist_group = []; all_wind_group = []

    print(f"\n=== Testing {NUM_TESTS} random scenarios ===")
    print(f"Candidates: {NUM_CANDIDATES}   Sampler: {SAMPLER} ({DDIM_STEPS} steps if ddim)   CFG: {CFG_SCALE}")
    print(f"{'#':>3s}  {'Init':>12s}  {'Target':>12s}  {'Error':>8s}  {'Speed':>6s}  {'Energy':>7s}  {'Result':>6s}")
    print("-" * 67)

    for i in range(NUM_TESTS):
        wx, wy, wvx, wvy = wind_buffer.get_list()
        # Random start index for wind (not always 0)
        max_start = len(wx) - 1 - max(config.TRAJ_LEN - 1, max(WIND_FRAME_OFFSETS))
        wind_start = random.randint(0, max(int(max_start), 1))

        while True:
            init_pos = [random.randint(20, config.WIDTH - 20),
                        random.randint(20, config.HEIGHT - 20)]
            target_pos = [random.randint(30, config.WIDTH - 30),
                          random.randint(30, config.HEIGHT - 30)]
            dist = math.sqrt((init_pos[0]-target_pos[0])**2 + (init_pos[1]-target_pos[1])**2)
            if dist > 50:
                break

        cond_np = np.array([
            init_pos[0]/config.WIDTH, init_pos[1]/config.HEIGHT,
            0.0, 0.0, config.START_TIME/24.0, config.START_ENERGY,
            target_pos[0]/config.WIDTH, target_pos[1]/config.HEIGHT,
        ], dtype=np.float32)
        cond = torch.tensor(cond_np).unsqueeze(0).to(device)

        # Route distance group (km)
        route_pix = math.sqrt((init_pos[0]-target_pos[0])**2 + (init_pos[1]-target_pos[1])**2)
        route_km = route_pix / config.WIDTH * config.WIDTH_KM
        if route_km < 150: dist_group = "short(<150km)"
        elif route_km < 300: dist_group = "mid(150-300)"
        else: dist_group = "long(>300km)"

        # 8-frame wind with random starting offset
        frame_indices = [wind_start + o for o in WIND_FRAME_OFFSETS]
        wind_u = torch.stack([wx[idx].float() for idx in frame_indices], dim=1).squeeze(2).to(device)
        wind_v = torch.stack([wy[idx].float() for idx in frame_indices], dim=1).squeeze(2).to(device)

        with torch.no_grad():
            all_trajs, all_dist_t, best = diffusion.sample_multi(
                cond, wind_u, wind_v, guidance_scale=CFG_SCALE, num_samples=NUM_CANDIDATES,
                sampler=SAMPLER, ddim_steps=DDIM_STEPS)
        traj_np = best[0].cpu().numpy()
        all_trajs_np = all_trajs[0].cpu().numpy()

        ex = traj_np[0, -1] * config.WIDTH
        ey = traj_np[1, -1] * config.HEIGHT
        d_pix = math.sqrt((ex-target_pos[0])**2 + (ey-target_pos[1])**2)
        d_km = d_pix / config.WIDTH * config.WIDTH_KM
        distances_km.append(d_km)
        success = d_km <= config.TARGET_RANGE
        if success: success_count += 1

        avg_speed, energy_end, path_len = compute_metrics(traj_np)

        all_init.append(init_pos); all_target.append(target_pos)
        all_dist.append(d_km); all_success.append(success)
        all_avg_speed.append(avg_speed); all_energy.append(energy_end)
        all_path_len.append(path_len)
        all_dist_group.append(dist_group)

        # Wind vs route direction group at start point
        wind_u_frame = wx[frame_indices[0]].squeeze().numpy()
        wind_v_frame = wy[frame_indices[0]].squeeze().numpy()
        gi = min(wind_u_frame.shape[0] - 1, max(0, int(init_pos[1] / config.RANGE_SCALE)))
        gj = min(wind_u_frame.shape[1] - 1, max(0, int(init_pos[0] / config.RANGE_SCALE)))
        w_vec = np.array([wind_u_frame[gi, gj], wind_v_frame[gi, gj]])
        r_vec = np.array([target_pos[0] - init_pos[0], target_pos[1] - init_pos[1]])
        cosang = float(w_vec @ r_vec / (np.linalg.norm(w_vec) * np.linalg.norm(r_vec) + 1e-6))
        if cosang > 0.3: wind_group = "tailwind"
        elif cosang < -0.3: wind_group = "headwind"
        else: wind_group = "crosswind"
        all_wind_group.append(wind_group)

        # Wind quiver uses the first wind frame in the sampled set

        status = "OK" if success else "FAIL"
        img_name = f"test_{i:03d}_{status}_{d_km:.0f}km.png"
        plot_multi_trajectory(all_trajs_np, traj_np, init_pos, target_pos,
                              wind_u_frame, wind_v_frame, d_km, success,
                              os.path.join(img_dir, img_name), i)

        print(f"{i:3d}  ({init_pos[0]:3d},{init_pos[1]:3d})  ({target_pos[0]:3d},{target_pos[1]:3d})  "
              f"{d_km:6.1f}km  {avg_speed:5.1f}m/s  {energy_end*100:5.0f}%   {status}")

    print("-" * 67)
    print(f"\n=== Summary ===")
    print(f"Tests: {NUM_TESTS}")
    print(f"Success rate: {success_count}/{NUM_TESTS} = {success_count/NUM_TESTS*100:.1f}%")
    print(f"Mean error: {np.mean(distances_km):.1f} km  Median: {np.median(distances_km):.1f} km")
    print(f"Min error: {np.min(distances_km):.1f} km   Max error: {np.max(distances_km):.1f} km")
    print(f"Mean speed: {np.mean(all_avg_speed):.1f} m/s")
    print(f"Mean remaining energy: {np.mean(all_energy)*100:.0f}%")
    print(f"Mean path length: {np.mean(all_path_len):.0f} km")
    group_report("Route distance groups", all_dist_group,
                 ["short(<150km)", "mid(150-300)", "long(>300km)"],
                 all_dist, all_success, all_avg_speed, all_energy, all_path_len)
    group_report("Wind direction groups", all_wind_group,
                 ["tailwind", "crosswind", "headwind"],
                 all_dist, all_success, all_avg_speed, all_energy, all_path_len)
    print(f"\nImages saved to: {img_dir}")

if __name__ == "__main__":
    main()




