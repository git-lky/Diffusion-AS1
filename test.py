"""Test: sample trajectories and evaluate with airship dynamics."""
import os, sys, math, random, time
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
from dynamics.airship import ASUnit_low
from dynamics.wind_field import WindGenerator
from dynamics.memory import Memory_wind_field

def load_model(checkpoint_path, device):
    unet = TrajectoryDiT().to(device)
    wind_encoder = (TemporalWindEncoder if config.WIND_NUM_FRAMES > 4 else WindEncoder)().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    unet.load_state_dict(ckpt["unet"])
    wind_encoder.load_state_dict(ckpt["wind_encoder"])
    unet.eval()
    wind_encoder.eval()
    diffusion = TrajectoryDiffusion(unet, wind_encoder).to(device)
    return diffusion

def main():
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

    # Random start/goal
    init_pos = [random.randint(20, config.WIDTH - 20), random.randint(20, config.HEIGHT - 20)]
    target_pos = [random.randint(30, config.WIDTH - 30), random.randint(30, config.HEIGHT - 30)]

    print(f"Init: {init_pos}, Target: {target_pos}")

    # Prepare condition
    cond = np.array([
        init_pos[0] / config.WIDTH, init_pos[1] / config.HEIGHT,
        0.0, 0.0, config.START_TIME / 24.0, config.START_ENERGY,
        target_pos[0] / config.WIDTH, target_pos[1] / config.HEIGHT,
    ], dtype=np.float32)
    cond = torch.tensor(cond).unsqueeze(0).to(device)

    # Get wind tensors for the first time step
    # 4-frame wind
    wind_u = torch.stack([wx[i].float() for i in [0,16,32,48]], dim=1).squeeze(2).to(device)
    wind_v = torch.stack([wy[i].float() for i in [0,16,32,48]], dim=1).squeeze(2).to(device)

    # Sample trajectory
    with torch.no_grad():
        traj = diffusion.sample(cond, wind_u, wind_v)
        traj_np = traj[0].cpu().numpy()  # (STATE_DIM, T) -> (6, 64)
    print(f"Sampled trajectory shape: {traj_np.shape}")

    # Denormalize and plot
    xs = traj_np[0] * config.WIDTH
    ys = traj_np[1] * config.HEIGHT

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, ys, 'b-', linewidth=1.5, label="Diffusion plan")
    ax.plot(init_pos[0], init_pos[1], 'go', markersize=10, label="Start")
    ax.plot(target_pos[0], target_pos[1], 'r*', markersize=15, label="Target")
    ax.set_xlim(0, config.WIDTH)
    ax.set_ylim(0, config.HEIGHT)
    ax.set_xlabel("X (pixel)"); ax.set_ylabel("Y (pixel)")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title("Diffusion-based Trajectory Planning")

    save_path = os.path.join(config.MODEL_SAVE_DIR, "trajectory_sample.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")

    # Evaluate: how close does the trajectory endpoint get to target?
    end_x, end_y = xs[-1], ys[-1]
    dist = math.sqrt((end_x - target_pos[0])**2 + (end_y - target_pos[1])**2)
    dist_km = dist / config.WIDTH * config.WIDTH_KM
    print(f"Endpoint distance to target: {dist:.1f} pixels ({dist_km:.1f} km)")

if __name__ == "__main__":
    main()
