"""Batch test: evaluate diffusion model + save trajectory plots."""
import os, sys, math, random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models.dit import TrajectoryDiT
from models.wind_encoder import WindEncoder, TemporalWindEncoder
from models.diffusion import TrajectoryDiffusion
from dynamics.wind_field import WindGenerator
from dynamics.memory import Memory_wind_field

def load_model(checkpoint_path, device):
    unet = TrajectoryDiT().to(device)
    wind_encoder = (TemporalWindEncoder if config.WIND_NUM_FRAMES > 4 else WindEncoder)().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    unet.load_state_dict(ckpt["unet"])
    wind_encoder.load_state_dict(ckpt["wind_encoder"])
    unet.eval(); wind_encoder.eval()
    return TrajectoryDiffusion(unet, wind_encoder).to(device)

def plot_trajectory(traj_np, init_pos, target_pos, d_km, success, save_path):
    xs = traj_np[0] * config.WIDTH
    ys = traj_np[1] * config.HEIGHT

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, 'b-', linewidth=1.2, alpha=0.8, label="Plan")
    ax.plot(init_pos[0], init_pos[1], 'go', markersize=8, label="Start")
    ax.plot(target_pos[0], target_pos[1], 'r*', markersize=12, label="Target")

    # Draw arrival circle
    r_pix = config.TARGET_RANGE / config.WIDTH_KM * config.WIDTH
    circle = plt.Circle((target_pos[0], target_pos[1]), r_pix,
                        color='r', fill=False, linestyle='--', linewidth=0.8, alpha=0.5)
    ax.add_patch(circle)

    # Mark endpoint
    ax.plot(xs[-1], ys[-1], 's', color='orange', markersize=6, label="End")

    status = "OK" if success else "FAIL"
    ax.set_title(f"Test #{test_id}: {status} | Error: {d_km:.1f} km")
    ax.set_xlim(0, config.WIDTH); ax.set_ylim(0, config.HEIGHT)
    ax.set_xlabel("X (pixel)"); ax.set_ylabel("Y (pixel)")
    ax.legend(loc='upper right', fontsize=7); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(config.MODEL_SAVE_DIR, "model_final.pt")
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint at {ckpt_path}")
        return
    diffusion = load_model(ckpt_path, device)
    print(f"Model loaded ({device})")

    # Output directory for images
    img_dir = os.path.join(config.MODEL_SAVE_DIR, "batch_results")
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

    NUM_TESTS = 30
    distances_km = []
    success_count = 0
    # Store scenarios for post-analysis
    all_init = []
    all_target = []
    all_dist = []
    all_success = []

    print(f"\n=== Testing {NUM_TESTS} random scenarios ===")
    print(f"Target arrival radius: {config.TARGET_RANGE} km")
    print(f"{'#':>3s}  {'Init':>12s}  {'Target':>12s}  {'Dist(km)':>10s}  {'Result':>8s}")
    print("-" * 55)

    for i in range(NUM_TESTS):
        global test_id
        test_id = i
        wx, wy, wvx, wvy = wind_buffer.get_list()
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
        # 4-frame wind for DiT
        wind_u = torch.stack([wx[i].float() for i in [0,16,32,48]], dim=1).squeeze(2).to(device)
        wind_v = torch.stack([wy[i].float() for i in [0,16,32,48]], dim=1).squeeze(2).to(device)

        with torch.no_grad():
            traj = diffusion.sample_multi_pick_best(cond, wind_u, wind_v, guidance_scale=2.0, num_samples=20)
        traj_np = traj[0].cpu().numpy()

        ex = traj_np[0, -1] * config.WIDTH
        ey = traj_np[1, -1] * config.HEIGHT
        d_pix = math.sqrt((ex-target_pos[0])**2 + (ey-target_pos[1])**2)
        d_km = d_pix / config.WIDTH * config.WIDTH_KM
        distances_km.append(d_km)
        success = d_km <= config.TARGET_RANGE
        if success: success_count += 1

        all_init.append(init_pos)
        all_target.append(target_pos)
        all_dist.append(d_km)
        all_success.append(success)

        # Save image for EVERY test
        status = "OK" if success else "FAIL"
        img_name = f"test_{i:03d}_{status}_{d_km:.0f}km.png"
        plot_trajectory(traj_np, init_pos, target_pos, d_km, success,
                       os.path.join(img_dir, img_name))

        print(f"{i:3d}  ({init_pos[0]:3d},{init_pos[1]:3d})  ({target_pos[0]:3d},{target_pos[1]:3d})  {d_km:10.1f}  {status}")

    print("-" * 55)

    # === Summary image: 3x3 grid of best cases ===
    sorted_idx = np.argsort(all_dist)
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    for j, idx in enumerate(sorted_idx[:9]):
        ax = axes[j//3][j%3]
        init = all_init[idx]; target = all_target[idx]
        d = all_dist[idx]; s = all_success[idx]
        # Re-sample for the plot (trajectory already generated above - need to redo)
        # Instead, just show the stored info
        ax.text(0.5, 0.5, f"#{idx}\n{init[0]},{init[1]}->{target[0]},{target[1]}\n{d:.0f} km {'OK' if s else 'FAIL'}",
                transform=ax.transAxes, ha='center', va='center', fontsize=8)
        ax.set_title(f"#{idx}: {d:.0f} km", fontsize=9)
        ax.axis('off')
    fig.suptitle(f"Best 9 Cases | Success Rate: {success_count}/{NUM_TESTS} ({success_count/NUM_TESTS*100:.0f}%)",
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(img_dir, "_summary_best9.png"), dpi=120)
    plt.close()

    print(f"\n=== Summary ===")
    print(f"Tests: {NUM_TESTS}")
    print(f"Success rate: {success_count}/{NUM_TESTS} = {success_count/NUM_TESTS*100:.1f}%")
    print(f"Mean error: {np.mean(distances_km):.1f} km")
    print(f"Median error: {np.median(distances_km):.1f} km")
    print(f"Min error: {np.min(distances_km):.1f} km")
    print(f"Max error: {np.max(distances_km):.1f} km")
    print(f"\nImages saved to: {img_dir}")

if __name__ == "__main__":
    main()
