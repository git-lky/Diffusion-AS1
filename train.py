"""Train trajectory diffusion model."""
import os, sys, time
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models.dit import TrajectoryDiT
from models.wind_encoder import WindEncoder, TemporalWindEncoder
from models.diffusion import TrajectoryDiffusion
from data.dataset import TrajectoryDataset

def main():
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    writer = SummaryWriter(config.LOG_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    dataset = TrajectoryDataset()
    dataloader = dataset.get_dataloader(batch_size=config.BATCH_SIZE, device=device)

    # Models
    unet = TrajectoryDiT().to(device)
    wind_encoder = (TemporalWindEncoder if config.WIND_NUM_FRAMES > 4 else WindEncoder)().to(device)
    diffusion = TrajectoryDiffusion(unet, wind_encoder).to(device)

    # EMA
    ema_unet = TrajectoryDiT().to(device)
    ema_wind_enc = (TemporalWindEncoder if config.WIND_NUM_FRAMES > 4 else WindEncoder)().to(device)
    ema_unet.load_state_dict(unet.state_dict())
    ema_wind_enc.load_state_dict(wind_encoder.state_dict())
    ema_diffusion = TrajectoryDiffusion(ema_unet, ema_wind_enc).to(device)

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=config.LEARNING_RATE)
    total_steps = 0

    for epoch in range(config.NUM_EPOCHS):
        epoch_loss = 0.0
        start_t = time.time()

        for batch_idx, (trajs, conditions, wind_u, wind_v) in enumerate(dataloader):
            trajs = trajs.to(device)
            conditions = conditions.to(device)

            # trajs: (B, T, STATE_DIM) -> (B, STATE_DIM, T) for Conv1d
            trajs = trajs.permute(0, 2, 1)

            # Split condition: first 6 = start_state, last 2 = goal_pos
            cond = conditions[:, :8]
            # For wind we need wind_u and wind_v tensors - use dummy for standalone training
            # In real use, wind comes from WindParser
            wind_u = wind_u.to(device)
            wind_v = wind_v.to(device)

            loss = diffusion.loss(trajs, cond, wind_u, wind_v)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), config.GRAD_CLIP)
            optimizer.step()

            # EMA update
            with torch.no_grad():
                for ema_p, p in zip(ema_diffusion.parameters(), diffusion.parameters()):
                    ema_p.data.lerp_(p.data, 1.0 - config.EMA_DECAY)

            epoch_loss += loss.item()
            writer.add_scalar("train/loss", loss.item(), total_steps)
            total_steps += 1

        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start_t
        print(f"Epoch {epoch:3d} | loss={avg_loss:.6f} | time={elapsed:.1f}s")

        writer.add_scalar("epoch/loss", avg_loss, epoch)

        if (epoch + 1) % config.SAVE_EVERY == 0:
            ckpt_path = os.path.join(config.MODEL_SAVE_DIR, f"checkpoint_{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "unet": unet.state_dict(),
                "wind_encoder": wind_encoder.state_dict(),
                "ema_unet": ema_unet.state_dict(),
                "ema_wind_enc": ema_wind_enc.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(config.MODEL_SAVE_DIR, "model_final.pt")
    torch.save({"unet": ema_unet.state_dict(), "wind_encoder": ema_wind_enc.state_dict()}, final_path)
    print(f"Final model saved: {final_path}")
    writer.close()

if __name__ == "__main__":
    main()
