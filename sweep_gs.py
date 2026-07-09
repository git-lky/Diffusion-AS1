"""Sweep guidance_scale to find optimal CFG value."""
import sys, os, math, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models.dit import TrajectoryDiT
from models.wind_encoder import WindEncoder, TemporalWindEncoder
from models.diffusion import TrajectoryDiffusion
from dynamics.wind_field import WindGenerator
from dynamics.memory import Memory_wind_field

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet = TrajectoryDiT().to(device)
we = WindEncoder().to(device)
diff = TrajectoryDiffusion(unet, we).to(device)
ckpt = torch.load(os.path.join(config.MODEL_SAVE_DIR, "model_final.pt"), map_location=device)
unet.load_state_dict(ckpt["unet"])
we.load_state_dict(ckpt["wind_encoder"])
print(f"Model loaded ({device})")

wg = WindGenerator(config.WIDTH, config.HEIGHT, 6, 10, 8, 8, 10, 10, 3, 10)
wb = Memory_wind_field(5)
for _ in range(5):
    wx, wy, wvx, wvy = wg.generate_wind_vector_list(120)
    wb.add_list(wx, wy, wvx, wvy)

print(f"\n{'GS':>6s}  {'OK':>6s}  {'Mean(km)':>10s}  {'Median(km)':>10s}")
print("-" * 40)

for gs in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    dists = []
    for _ in range(30):
        wx, wy, wvx, wvy = wb.get_list()
        while True:
            ip = [random.randint(20, config.WIDTH - 20),
                  random.randint(20, config.HEIGHT - 20)]
            tp = [random.randint(30, config.WIDTH - 30),
                  random.randint(30, config.HEIGHT - 30)]
            if math.sqrt((ip[0]-tp[0])**2 + (ip[1]-tp[1])**2) > 50:
                break
        c = torch.tensor([
            ip[0]/config.WIDTH, ip[1]/config.HEIGHT,
            0, 0, config.START_TIME/24, config.START_ENERGY,
            tp[0]/config.WIDTH, tp[1]/config.HEIGHT
        ]).unsqueeze(0).to(device)
        # 4-frame wind
        wu = torch.stack([wx[i].float() for i in [0, 16, 32, 48]], dim=1).squeeze(2).to(device)
        wv = torch.stack([wy[i].float() for i in [0, 16, 32, 48]], dim=1).squeeze(2).to(device)
        traj = diff.sample_with_guidance(c, wu, wv, guidance_scale=gs)
        ex = traj[0, 0, -1].item() * config.WIDTH
        ey = traj[0, 1, -1].item() * config.HEIGHT
        d = math.sqrt((ex-tp[0])**2 + (ey-tp[1])**2) / config.WIDTH * config.WIDTH_KM
        dists.append(d)
    ok = sum(1 for d in dists if d <= config.TARGET_RANGE)
    print(f"{gs:6.1f}  {ok:2d}/30  {np.mean(dists):10.1f}  {np.median(dists):10.1f}")
