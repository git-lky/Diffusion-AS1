"""Test: sample multiple trajectories with wind overlay and evaluate."""
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
from batch_test import plot_multi_trajectory, compute_metrics

NUM_CANDIDATES = 5

def parse_args():
    p = argparse.ArgumentParser(description="Single-scenario test with multi-trajectory plot")
    p.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    p.add_argument("--steps", type=int, default=50, help="DDIM steps (used when --sampler ddim)")
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--candidates", type=int, default=5)
    return p.parse_args()

def load_model(checkpoint_path, device):
    dit = create_denoiser().to(device)
    wind_encoder = (TemporalWindEncoder if config.WIND_NUM_FRAMES > 4 else WindEncoder)().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    dit.load_state_dict(ckpt["dit"])
    wind_encoder.load_state_dict(ckpt["wind_encoder"])
    dit.eval()
    wind_encoder.eval()
    diffusion = TrajectoryDiffusion(dit, wind_encoder).to(device)
    return diffusion

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(config.MODEL_SAVE_DIR, "model_final.pt")

    if not os.path.exists(ckpt_path):
        print(f"No checkpoint at {ckpt_path}. Run training first.")
        return

    diffusion = load_model(ckpt_path, device)
    print("Model loaded.")

    # Generate wind field for test
    wind_gen = WindGenerator(
        config.WIDTH, config.HEIGHT,
        config.WIND_MIN_AVE_V, config.WIND_MAX_AVE_V,
        config.WIND_VARY_SPEED, config.WIND_VARY_NUM,
        10, 10, config.WIND_NUM_INP, config.RANGE_SCALE
    )
    wind_buffer = Memory_wind_field(1)
    wx, wy, wvx, wvy = wind_gen.generate_wind_vector_list(config.WIND_HOURS)
    wind_buffer.add_list(wx, wy, wvx, wvy)
    wx, wy, wvx, wvy = wind_buffer.get_list()

    # Random wind start index (same as batch_test)
    max_start = len(wx) - max(config.WIND_FRAME_OFFSETS) - 1
    wind_start = random.randint(0, max(int(max_start), 1))

    # Random start/goal
    init_pos = [random.randint(20, config.WIDTH - 20), random.randint(20, config.HEIGHT - 20)]
    target_pos = [random.randint(30, config.WIDTH - 30), random.randint(30, config.HEIGHT - 30)]

    print(f"Init: {init_pos}, Target: {target_pos}, Wind start frame: {wind_start}")

    # Prepare condition
    cond = np.array([
        init_pos[0] / config.WIDTH, init_pos[1] / config.HEIGHT,
        0.0, 0.0, config.START_TIME / 24.0, config.START_ENERGY,
        target_pos[0] / config.WIDTH, target_pos[1] / config.HEIGHT,
    ], dtype=np.float32)
    cond = torch.tensor(cond).unsqueeze(0).to(device)

    # Wind frames at random start
    frame_indices = [wind_start + o for o in config.WIND_FRAME_OFFSETS]
    wind_u = torch.stack([wx[i].float() for i in frame_indices], dim=1).squeeze(2).to(device)
    wind_v = torch.stack([wy[i].float() for i in frame_indices], dim=1).squeeze(2).to(device)

    # Sample multiple trajectories and pick the best
    with torch.no_grad():
        all_trajs, dists, best = diffusion.sample_multi(
            cond, wind_u, wind_v, guidance_scale=args.cfg, num_samples=args.candidates,
            sampler=args.sampler, ddim_steps=args.steps)
    all_trajs_np = all_trajs[0].cpu().numpy()
    best_np = best[0].cpu().numpy()

    # Metrics
    avg_speed, energy_end, path_len = compute_metrics(best_np)
    end_x, end_y = best_np[0, -1] * config.WIDTH, best_np[1, -1] * config.HEIGHT
    d_pix = math.sqrt((end_x - target_pos[0]) ** 2 + (end_y - target_pos[1]) ** 2)
    d_km = d_pix / config.WIDTH * config.WIDTH_KM
    success = d_km <= config.TARGET_RANGE

    print(f"Candidates: {args.candidates} | sampler: {args.sampler} | best error: {d_km:.1f} km | "
          f"avg speed: {avg_speed:.1f} m/s | energy: {energy_end * 100:.0f}% | path: {path_len:.0f} km")

    # Plot all candidates + best + wind quiver
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(config.MODEL_SAVE_DIR, "trajectory_sample.png")
    wind_u_frame = wx[frame_indices[0]].squeeze().numpy()
    wind_v_frame = wy[frame_indices[0]].squeeze().numpy()
    plot_multi_trajectory(all_trajs_np, best_np, init_pos, target_pos,
                          wind_u_frame, wind_v_frame, d_km, success, save_path, 0)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()
