"""Train the one-hour, 7.5-minute-replanning trajectory diffusion model."""
import json
import hashlib
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data.dataset import TrajectoryDataset
from dynamics.wind_field import WindGenerator
from models import create_denoiser
from models.diffusion import TrajectoryDiffusion
from models.wind_encoder import SpatioTemporalWindEncoder
from dynamics.rolling import (
    distance_km_torch,
    sample_wind_numpy,
    target_from_condition_torch,
)
from rolling_planner import run_rolling_planner


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__), text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def working_tree_hash():
    """Fingerprint tracked source changes so a dirty-tree run remains identifiable."""
    try:
        diff = subprocess.check_output(
            ["git", "diff", "--no-ext-diff", "HEAD", "--", "*.py"],
            cwd=os.path.dirname(__file__),
        )
        return hashlib.sha256(diff).hexdigest()[:16]
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def checkpoint_metadata():
    metadata = config.model_metadata()
    metadata["git_commit"] = git_commit()
    metadata["working_tree_hash"] = working_tree_hash()
    return metadata


def save_checkpoint(path, epoch, total_steps, avg_loss, dit, wind_encoder,
                    ema_dit, ema_wind_enc, optimizer, validation_loss=None,
                    mission_score=None, mission_metrics=None,
                    mission_checks_without_improvement=0):
    state = {
        "metadata": checkpoint_metadata(),
        "epoch": epoch,
        "total_steps": total_steps,
        "dit": dit.state_dict(),
        "wind_encoder": wind_encoder.state_dict(),
        "ema_dit": ema_dit.state_dict(),
        "ema_wind_enc": ema_wind_enc.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": avg_loss,
        "validation_loss": validation_loss,
        "mission_score": mission_score,
        "mission_metrics": mission_metrics,
        "mission_checks_without_improvement": mission_checks_without_improvement,
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    temporary = path + ".tmp"
    torch.save(state, temporary)
    os.replace(temporary, path)


def validate_metadata(checkpoint):
    expected = config.model_metadata()
    actual = checkpoint.get("metadata")
    if actual is None:
        raise RuntimeError(
            "checkpoint has no rolling-DiT metadata; use checkpoints_rolling, not the old checkpoint"
        )
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"checkpoint/config mismatch: {json.dumps(mismatches, ensure_ascii=False)}")


def move_batch(batch, device):
    controls, conditions, wind_u, wind_v, states, valid, previous, previous_valid = batch
    return (
        controls.to(device),
        conditions.to(device),
        wind_u.to(device, dtype=torch.float32),
        wind_v.to(device, dtype=torch.float32),
        states.to(device),
        valid.to(device),
        previous.to(device),
        previous_valid.to(device),
    )


def evaluate(diffusion, dataloader, device):
    diffusion.eval()
    totals = []
    with torch.no_grad():
        for batch in dataloader:
            controls, cond, wind_u, wind_v, states, valid, previous, previous_valid = move_batch(
                batch, device
            )
            losses = diffusion.loss(
                controls.permute(0, 2, 1), cond, wind_u, wind_v,
                states.permute(0, 2, 1), valid,
                previous.permute(0, 2, 1), previous_valid,
            )
            totals.append(float(losses["total"]))
    diffusion.train()
    return sum(totals) / max(len(totals), 1)


@torch.no_grad()
def evaluate_generation(diffusion, dataloader, device):
    """Validate free-running cold-start samples, not only denoising loss."""
    diffusion.eval()
    batch = next(iter(dataloader))
    _, cond, wind_u, wind_v, _, _, _, _ = move_batch(batch, device)
    count = min(config.GENERATION_VALIDATE_SAMPLES, cond.shape[0])
    cond = cond[:count]
    wind_u = wind_u[:count]
    wind_v = wind_v[:count]
    cold_start = torch.zeros(
        count, config.ACTION_DIM, config.TRAJ_LEN, device=device, dtype=cond.dtype
    )
    sampled = diffusion.sample_multi(
        cond, wind_u, wind_v, cold_start,
        guidance_scale=1.5, num_samples=3, sampler="ddim", ddim_steps=25,
        physics_guidance_scale=0.0,
    )
    states = sampled["best_states"]
    target = target_from_condition_torch(cond)
    start_distance = distance_km_torch(cond[:, :2], target)
    end_distance = distance_km_torch(states[:, :2, -1], target)
    progress = start_distance - end_distance
    outside = (
        (states[:, 0] < 0.0) | (states[:, 0] > 1.0)
        | (states[:, 1] < 0.0) | (states[:, 1] > 1.0)
    ).any(dim=1)
    return {
        "mean_progress_km": float(progress.mean()),
        "progress_fraction": float((progress > 0.0).float().mean()),
        "boundary_fraction": float(outside.float().mean()),
        "feasible_fraction": float(sampled["feasible"].any(dim=1).float().mean()),
    }


def _wind_relation(wind_u, wind_v, frame, start, goal):
    """Use the same route-window wind classification as batch evaluation."""
    scale = np.array([config.WIDTH, config.HEIGHT], dtype=np.float32)
    start_norm = np.asarray(start, dtype=np.float32) / scale
    goal_norm = np.asarray(goal, dtype=np.float32) / scale
    route = (goal_norm - start_norm) * np.array(
        [config.WIDTH_KM, config.HEIGHT_KM], dtype=np.float32
    )
    winds = []
    for offset in range(config.TRAJ_LEN):
        fraction = (offset + 1) / config.TRAJ_LEN
        position = start_norm + fraction * (goal_norm - start_norm)
        winds.append([
            sample_wind_numpy(wind_u[frame + offset].squeeze(0).numpy(), position),
            sample_wind_numpy(wind_v[frame + offset].squeeze(0).numpy(), position),
        ])
    wind = np.mean(np.asarray(winds, dtype=np.float32), axis=0)
    cosine = float(wind @ route / (np.linalg.norm(wind) * np.linalg.norm(route) + 1e-6))
    return "tailwind" if cosine > 0.3 else "headwind" if cosine < -0.3 else "crosswind"


def build_fixed_mission_suite():
    """Create reproducible unseen closed-loop tasks without altering training RNG state."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random.seed(config.RANDOM_SEED + 701)
    np.random.seed(config.RANDOM_SEED + 701)
    generator = WindGenerator(
        config.WIDTH, config.HEIGHT,
        config.WIND_MIN_AVE_V, config.WIND_MAX_AVE_V,
        config.WIND_VARY_SPEED, config.WIND_VARY_NUM,
        10, 10, config.WIND_NUM_INP, config.RANGE_SCALE,
    )
    desired = [
        # Eight medium-distance headwind tasks target the v5 failure cluster;
        # four controls retain crosswind/tailwind coverage against regression.
        ("headwind", 150.0, 300.0, 0.0, 0.65),
        ("headwind", 150.0, 300.0, 3.0, 0.75),
        ("headwind", 150.0, 300.0, 6.0, 0.85),
        ("headwind", 150.0, 300.0, 9.0, 0.70),
        ("headwind", 150.0, 300.0, 12.0, 0.80),
        ("headwind", 150.0, 300.0, 15.0, 0.90),
        ("headwind", 150.0, 300.0, 18.0, 0.70),
        ("headwind", 150.0, 300.0, 21.0, 0.80),
        ("crosswind", 150.0, 300.0, 5.0, 0.70),
        ("crosswind", 150.0, 300.0, 17.0, 0.85),
        ("tailwind", 150.0, 300.0, 8.0, 0.70),
        ("tailwind", 150.0, 300.0, 20.0, 0.85),
    ][:config.FULL_MISSION_VALIDATE_SAMPLES]
    required = config.MAX_MISSION_STEPS + config.WIND_NUM_FRAMES + 1
    scenarios = []
    for relation, minimum, maximum, start_time, start_energy in desired:
        wind_u, wind_v, _, _ = generator.generate_wind_vector_list(config.WIND_HOURS)
        wind_start = random.randint(0, len(wind_u) - required)
        chosen = None
        for _ in range(5000):
            start = [
                random.randint(20, config.WIDTH - 20),
                random.randint(20, config.HEIGHT - 20),
            ]
            goal = [
                random.randint(30, config.WIDTH - 30),
                random.randint(30, config.HEIGHT - 30),
            ]
            dx = (start[0] - goal[0]) / config.WIDTH * config.WIDTH_KM
            dy = (start[1] - goal[1]) / config.HEIGHT * config.HEIGHT_KM
            distance = math.hypot(dx, dy)
            if minimum <= distance <= maximum and _wind_relation(
                wind_u, wind_v, wind_start, start, goal
            ) == relation:
                chosen = (start, goal)
                break
        if chosen is None:
            raise RuntimeError(f"could not construct fixed {relation} validation task")
        scenarios.append({
            "wind_u": wind_u, "wind_v": wind_v, "wind_start": wind_start,
            "start": chosen[0], "goal": chosen[1], "start_time": start_time,
            "start_energy": start_energy, "group": relation,
        })
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    return scenarios


@torch.no_grad()
def evaluate_closed_loop(diffusion, scenarios, device):
    """Evaluate the same complete missions at each model-selection point."""
    diffusion.eval()
    results = []
    for scenario in scenarios:
        result = run_rolling_planner(
            diffusion, scenario["wind_u"], scenario["wind_v"],
            scenario["start"], scenario["goal"],
            wind_start=scenario["wind_start"],
            start_time=scenario["start_time"],
            start_energy=scenario["start_energy"],
            guidance_scale=1.5,
            candidates=config.FULL_MISSION_VALIDATE_CANDIDATES,
            sampler="ddim", ddim_steps=config.FULL_MISSION_VALIDATE_DDIM_STEPS,
            physics_guidance_scale=0.0, device=device,
        )
        results.append(result)
    success = float(np.mean([result.success for result in results]))
    endpoint = float(np.mean([
        result.metrics["endpoint_error_km"] for result in results
    ]))
    final_energy = float(np.mean([
        result.metrics["final_energy"] for result in results
    ]))
    reset_fraction = float(np.mean([
        result.metrics["plan_reset_count"] / max(result.metrics["replans"], 1.0)
        for result in results
    ]))
    infeasible_fraction = float(np.mean([
        result.metrics["infeasible_plan_count"] / max(result.metrics["replans"], 1.0)
        for result in results
    ]))
    score = (
        5.0 * (1.0 - success)
        + endpoint / max(config.WIDTH_KM, config.HEIGHT_KM)
        + 0.25 * reset_fraction
        + 0.50 * infeasible_fraction
        + max(0.0, config.ENERGY_RESERVE_BASE - final_energy)
    )
    return score, {
        "success_fraction": success,
        "mean_endpoint_error_km": endpoint,
        "mean_final_energy": final_energy,
        "mean_plan_reset_fraction": reset_fraction,
        "mean_infeasible_plan_fraction": infeasible_fraction,
    }


def epoch_learning_rate(epoch):
    """Linear warmup followed by cosine decay, deterministic across resume."""
    if epoch < config.LR_WARMUP_EPOCHS:
        return config.LEARNING_RATE * (epoch + 1) / config.LR_WARMUP_EPOCHS
    denominator = max(config.NUM_EPOCHS - config.LR_WARMUP_EPOCHS - 1, 1)
    progress = (epoch - config.LR_WARMUP_EPOCHS) / denominator
    return config.LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    writer = SummaryWriter(config.LOG_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(
        f"Rolling plan: {config.PLAN_HOURS}h/{config.TRAJ_LEN} steps; "
        f"wind context {config.WIND_CONTEXT_HOURS}h/{config.WIND_NUM_FRAMES} frames; "
        f"execute {config.REPLAN_MINUTES}min/{config.EXECUTE_STEPS} steps; "
        f"train {config.NUM_EPOCHS} epochs"
    )

    dataset = TrajectoryDataset()
    train_loader = dataset.get_dataloader(batch_size=config.BATCH_SIZE, split="train")
    val_loader = dataset.get_dataloader(batch_size=config.BATCH_SIZE, split="val")
    mission_suite = build_fixed_mission_suite()
    print(
        "Fixed closed-loop validation suite: "
        + ", ".join(scenario["group"] for scenario in mission_suite)
    )

    dit = create_denoiser().to(device)
    wind_encoder = SpatioTemporalWindEncoder().to(device)
    diffusion = TrajectoryDiffusion(dit, wind_encoder).to(device)
    ema_dit = create_denoiser().to(device)
    ema_wind_encoder = SpatioTemporalWindEncoder().to(device)
    ema_dit.load_state_dict(dit.state_dict())
    ema_wind_encoder.load_state_dict(wind_encoder.state_dict())
    ema_diffusion = TrajectoryDiffusion(ema_dit, ema_wind_encoder).to(device)

    model_params = tuple(diffusion.parameters())
    ema_params = tuple(ema_diffusion.parameters())
    if len(model_params) != len(ema_params):
        raise RuntimeError("model and EMA parameter counts do not match")
    optimizer = torch.optim.AdamW(model_params, lr=config.LEARNING_RATE)

    start_epoch = 0
    total_steps = 0
    latest_path = os.path.join(config.MODEL_SAVE_DIR, "checkpoint_latest.pt")
    best_path = os.path.join(config.MODEL_SAVE_DIR, "checkpoint_best.pt")
    best_mission_path = os.path.join(
        config.MODEL_SAVE_DIR, "checkpoint_best_mission.pt"
    )
    best_validation = float("inf")
    best_mission_score = float("inf")
    mission_checks_without_improvement = 0
    if os.path.exists(best_path):
        best_state = torch.load(best_path, map_location="cpu", weights_only=False)
        validate_metadata(best_state)
        stored_best = best_state.get("validation_loss")
        if stored_best is not None:
            best_validation = float(stored_best)
    if os.path.exists(best_mission_path):
        best_mission_state = torch.load(
            best_mission_path, map_location="cpu", weights_only=False
        )
        validate_metadata(best_mission_state)
        stored_score = best_mission_state.get("mission_score")
        if stored_score is not None:
            best_mission_score = float(stored_score)
    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        validate_metadata(checkpoint)
        dit.load_state_dict(checkpoint["dit"])
        wind_encoder.load_state_dict(checkpoint["wind_encoder"])
        ema_dit.load_state_dict(checkpoint["ema_dit"])
        ema_wind_encoder.load_state_dict(checkpoint["ema_wind_enc"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        total_steps = int(checkpoint.get("total_steps", start_epoch * len(train_loader)))
        mission_checks_without_improvement = int(
            checkpoint.get("mission_checks_without_improvement", 0)
        )
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if "cuda_rng_state_all" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state_all"]])
        print(f"Resumed from {latest_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        stop_after_epoch = False
        learning_rate = epoch_learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        diffusion.train()
        epoch_components = {}
        start_time = time.time()
        for batch in train_loader:
            controls, cond, wind_u, wind_v, states, valid, previous, previous_valid = move_batch(
                batch, device
            )
            losses = diffusion.loss(
                controls.permute(0, 2, 1), cond, wind_u, wind_v,
                states.permute(0, 2, 1), valid,
                previous.permute(0, 2, 1), previous_valid,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model_params, config.GRAD_CLIP)
            optimizer.step()

            with torch.no_grad():
                for ema_parameter, parameter in zip(ema_params, model_params):
                    ema_parameter.data.lerp_(parameter.data, 1.0 - config.EMA_DECAY)

            for name, value in losses.items():
                epoch_components[name] = epoch_components.get(name, 0.0) + float(value)
                writer.add_scalar(f"train/{name}", float(value), total_steps)
            total_steps += 1

        averages = {name: value / max(len(train_loader), 1) for name, value in epoch_components.items()}
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:3d} | total={averages['total']:.6f} "
            f"diff={averages['diffusion']:.6f} state={averages['state']:.6f} "
            f"time={elapsed:.1f}s"
        )
        for name, value in averages.items():
            writer.add_scalar(f"epoch/{name}", value, epoch)
        writer.add_scalar("epoch/learning_rate", learning_rate, epoch)

        if (epoch + 1) % config.VALIDATE_EVERY == 0:
            val_loss = evaluate(ema_diffusion, val_loader, device)
            writer.add_scalar("validation/total", val_loss, epoch)
            print(f"Validation | total={val_loss:.6f}")
            if val_loss < best_validation:
                best_validation = val_loss
                save_checkpoint(
                    best_path, epoch, total_steps, averages["total"], dit, wind_encoder,
                    ema_dit, ema_wind_encoder, optimizer, validation_loss=val_loss,
                )
                print(f"Saved best checkpoint: {best_path}")

        if (epoch + 1) % config.GENERATION_VALIDATE_EVERY == 0:
            generated = evaluate_generation(ema_diffusion, val_loader, device)
            for name, value in generated.items():
                writer.add_scalar(f"generation/{name}", value, epoch)
            print(
                "Generation | "
                f"progress={generated['mean_progress_km']:.1f}km "
                f"positive={generated['progress_fraction']:.1%} "
                f"safe={1.0 - generated['boundary_fraction']:.1%} "
                f"feasible={generated['feasible_fraction']:.1%}"
            )
            generation_failures = []
            if generated["mean_progress_km"] < config.GENERATION_MIN_PROGRESS_KM:
                generation_failures.append("mean progress is too small")
            if generated["progress_fraction"] < config.GENERATION_MIN_POSITIVE_FRACTION:
                generation_failures.append("too few samples move toward the goal")
            if generated["boundary_fraction"] > config.GENERATION_MAX_BOUNDARY_FRACTION:
                generation_failures.append("too many generated trajectories cross the boundary")
            if generated["feasible_fraction"] < config.GENERATION_MIN_FEASIBLE_FRACTION:
                generation_failures.append("too few candidate sets contain a feasible plan")
            if generation_failures:
                save_checkpoint(
                    latest_path, epoch, total_steps, averages["total"], dit, wind_encoder,
                    ema_dit, ema_wind_encoder, optimizer,
                )
                raise RuntimeError(
                    "generation quality gate failed: " + "; ".join(generation_failures)
                )

        if (epoch + 1) % config.FULL_MISSION_VALIDATE_EVERY == 0:
            mission_score, mission_metrics = evaluate_closed_loop(
                ema_diffusion, mission_suite, device
            )
            for name, value in mission_metrics.items():
                writer.add_scalar(f"closed_loop/{name}", value, epoch)
            writer.add_scalar("closed_loop/selection_score", mission_score, epoch)
            print(
                "Closed loop | "
                f"success={mission_metrics['success_fraction']:.1%} "
                f"error={mission_metrics['mean_endpoint_error_km']:.1f}km "
                f"energy={mission_metrics['mean_final_energy']:.1%} "
                f"resets={mission_metrics['mean_plan_reset_fraction']:.1%} "
                f"infeasible={mission_metrics['mean_infeasible_plan_fraction']:.1%} "
                f"score={mission_score:.4f}"
            )
            if mission_score < best_mission_score:
                best_mission_score = mission_score
                mission_checks_without_improvement = 0
                save_checkpoint(
                    best_mission_path, epoch, total_steps, averages["total"],
                    dit, wind_encoder, ema_dit, ema_wind_encoder, optimizer,
                    mission_score=mission_score, mission_metrics=mission_metrics,
                )
                print(f"Saved best closed-loop checkpoint: {best_mission_path}")
            else:
                mission_checks_without_improvement += 1
            if (
                epoch + 1 >= config.EARLY_STOP_MIN_EPOCHS
                and mission_checks_without_improvement >= config.EARLY_STOP_PATIENCE
            ):
                stop_after_epoch = True
                print(
                    "Closed-loop early stop requested: "
                    f"no improvement for {mission_checks_without_improvement} checks"
                )

        if (epoch + 1) % config.RESUME_SAVE_EVERY == 0:
            save_checkpoint(
                latest_path, epoch, total_steps, averages["total"], dit, wind_encoder,
                ema_dit, ema_wind_encoder, optimizer,
                mission_checks_without_improvement=mission_checks_without_improvement,
            )
            print(f"Saved rolling checkpoint: {latest_path}")
        if (epoch + 1) % config.SAVE_EVERY == 0:
            path = os.path.join(config.MODEL_SAVE_DIR, f"checkpoint_{epoch + 1}.pt")
            save_checkpoint(
                path, epoch, total_steps, averages["total"], dit, wind_encoder,
                ema_dit, ema_wind_encoder, optimizer,
            )
            print(f"Saved checkpoint: {path}")
        if stop_after_epoch:
            break

    selected_path = best_mission_path if os.path.exists(best_mission_path) else best_path
    if os.path.exists(selected_path):
        best_state = torch.load(selected_path, map_location=device, weights_only=False)
        ema_dit.load_state_dict(best_state["ema_dit"])
        ema_wind_encoder.load_state_dict(best_state["ema_wind_enc"])
        if selected_path == best_mission_path:
            print(
                f"Using best closed-loop checkpoint from epoch "
                f"{int(best_state['epoch']) + 1} "
                f"(score={float(best_state['mission_score']):.6f})"
            )
        else:
            print(
                f"Using best validation checkpoint from epoch {int(best_state['epoch']) + 1} "
                f"(loss={float(best_state['validation_loss']):.6f})"
            )
    final_path = os.path.join(config.MODEL_SAVE_DIR, "model_final.pt")
    torch.save({
        "metadata": checkpoint_metadata(),
        "dit": ema_dit.state_dict(),
        "wind_encoder": ema_wind_encoder.state_dict(),
    }, final_path)
    print(f"Final model saved: {final_path}")
    writer.close()


if __name__ == "__main__":
    main()
