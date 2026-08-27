"""Train trajectory diffusion model."""
import os, sys, time
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models import create_denoiser
from models.wind_encoder import WindEncoder, TemporalWindEncoder
from models.diffusion import TrajectoryDiffusion
from data.dataset import TrajectoryDataset

def save_checkpoint(path, epoch, total_steps, avg_loss, dit, wind_encoder,
                    ema_dit, ema_wind_enc, optimizer):
    """Atomically save a resumable checkpoint."""
    state = {
        "epoch": epoch,
        "total_steps": total_steps,
        "dit": dit.state_dict(),
        "wind_encoder": wind_encoder.state_dict(),
        "ema_dit": ema_dit.state_dict(),
        "ema_wind_enc": ema_wind_enc.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": avg_loss,
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)

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
    dit = create_denoiser().to(device)
    wind_encoder = (TemporalWindEncoder if config.WIND_NUM_FRAMES > 4 else WindEncoder)().to(device)
    diffusion = TrajectoryDiffusion(dit, wind_encoder).to(device)

    # EMA
    ema_dit = create_denoiser().to(device)
    ema_wind_enc = (TemporalWindEncoder if config.WIND_NUM_FRAMES > 4 else WindEncoder)().to(device)
    ema_dit.load_state_dict(dit.state_dict())
    ema_wind_enc.load_state_dict(wind_encoder.state_dict())
    ema_diffusion = TrajectoryDiffusion(ema_dit, ema_wind_enc).to(device)

    # Materialize the parameter lists once. Besides avoiding repeated module-tree
    # traversal in every batch, this guarantees a stable one-to-one EMA mapping.
    model_params = tuple(diffusion.parameters())
    ema_params = tuple(ema_diffusion.parameters())
    if len(model_params) != len(ema_params):
        raise RuntimeError("Model and EMA parameter counts do not match")

    optimizer = torch.optim.AdamW(model_params, lr=config.LEARNING_RATE)
    start_epoch = 0
    total_steps = 0

    # Resume only from the rolling checkpoint created by this training series.
    # Historical checkpoint_*.pt files are deliberately not selected here.
    latest_path = os.path.join(config.MODEL_SAVE_DIR, "checkpoint_latest.pt")
    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        dit.load_state_dict(checkpoint["dit"])
        wind_encoder.load_state_dict(checkpoint["wind_encoder"])
        ema_dit.load_state_dict(checkpoint["ema_dit"])
        ema_wind_enc.load_state_dict(checkpoint["ema_wind_enc"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        total_steps = int(checkpoint.get("total_steps", start_epoch * len(dataloader)))
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if "cuda_rng_state_all" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        print(f"Resumed from {latest_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, config.NUM_EPOCHS):
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
            torch.nn.utils.clip_grad_norm_(model_params, config.GRAD_CLIP)
            optimizer.step()

            # EMA update
            with torch.no_grad():
                for ema_p, p in zip(ema_params, model_params):
                    ema_p.data.lerp_(p.data, 1.0 - config.EMA_DECAY)

            epoch_loss += loss.item()
            writer.add_scalar("train/loss", loss.item(), total_steps)
            total_steps += 1

        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start_t
        print(f"Epoch {epoch:3d} | loss={avg_loss:.6f} | time={elapsed:.1f}s")

        writer.add_scalar("epoch/loss", avg_loss, epoch)

        if (epoch + 1) % config.RESUME_SAVE_EVERY == 0:
            save_checkpoint(latest_path, epoch, total_steps, avg_loss, dit,
                            wind_encoder, ema_dit, ema_wind_enc, optimizer)
            print(f"Saved rolling checkpoint: {latest_path}")

        if (epoch + 1) % config.SAVE_EVERY == 0:
            ckpt_path = os.path.join(config.MODEL_SAVE_DIR, f"checkpoint_{epoch+1}.pt")
            save_checkpoint(ckpt_path, epoch, total_steps, avg_loss, dit,
                            wind_encoder, ema_dit, ema_wind_enc, optimizer)
            print(f"Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(config.MODEL_SAVE_DIR, "model_final.pt")
    torch.save({"dit": ema_dit.state_dict(), "wind_encoder": ema_wind_enc.state_dict()}, final_path)
    print(f"Final model saved: {final_path}")
    writer.close()

if __name__ == "__main__":
    main()


