"""DDPM/DDIM training and sampling for rolling airship control trajectories."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from dynamics.rolling import (
    candidate_cost_torch,
    distance_km_torch,
    required_energy_reserve_torch,
    rollout_controls_torch,
    target_from_condition_torch,
)


class TrajectoryDiffusion(nn.Module):
    def __init__(self, denoiser, wind_encoder, cfg_dropout=config.CFG_DROPOUT):
        super().__init__()
        self.denoiser = denoiser
        self.wind_encoder = wind_encoder
        betas = torch.tensor(config.cosine_beta_schedule(config.DIFFUSION_STEPS), dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.tensor([1.0]), alpha_bar[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alpha_bar)
        self.register_buffer("alphas_cumprod_prev", alpha_bar_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alpha_bar))
        self.register_buffer(
            "posterior_variance", betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        )

        self.num_steps = config.DIFFUSION_STEPS
        self.cfg_dropout = cfg_dropout
        self.action_dim = config.ACTION_DIM
        self.traj_len = config.TRAJ_LEN
        if config.PREDICTION_TYPE not in {"epsilon", "x0"}:
            raise ValueError("PREDICTION_TYPE must be 'epsilon' or 'x0'")
        self.prediction_type = config.PREDICTION_TYPE

    def q_sample(self, x0, t, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise
        a = self.sqrt_alphas_cumprod[t][:, None, None]
        b = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return a * x0 + b * noise, noise

    @staticmethod
    def unconditional_condition(cond):
        """Drop task intent for CFG while retaining the physically required state."""
        uncond = cond.clone()
        uncond[:, config.COND_GOAL_DX] = 0.0
        uncond[:, config.COND_GOAL_DY] = 0.0
        uncond[:, config.COND_REMAINING] = 0.0
        return uncond

    def _x0_from_output(self, xt, t, model_output):
        if self.prediction_type == "x0":
            return model_output
        alpha_bar = self.alphas_cumprod[t][:, None, None]
        return (xt - torch.sqrt(1.0 - alpha_bar) * model_output) / torch.sqrt(alpha_bar)

    def _epsilon_from_output(self, xt, t, model_output):
        if self.prediction_type == "epsilon":
            return model_output
        alpha_bar = self.alphas_cumprod[t][:, None, None]
        return (xt - torch.sqrt(alpha_bar) * model_output) / torch.sqrt(1.0 - alpha_bar)

    def _model_output(self, xt, t, cond, wind_tokens, previous_plan, guidance_scale):
        conditional = self.denoiser(xt, t, cond, wind_tokens, previous_plan)
        if not torch.is_tensor(guidance_scale) and guidance_scale == 1.0:
            return conditional
        unconditional = self.denoiser(
            xt, t, self.unconditional_condition(cond), wind_tokens, previous_plan
        )
        scale = torch.as_tensor(
            guidance_scale, device=xt.device, dtype=xt.dtype
        )
        if scale.ndim == 0:
            scale = scale.expand(xt.shape[0])
        return unconditional + scale[:, None, None] * (conditional - unconditional)

    def _ddpm_step(self, xt, t, model_output):
        x0 = self._x0_from_output(xt, t, model_output).clamp(-1.0, 1.0)
        alpha_bar = self.alphas_cumprod[t][:, None, None]
        noise_pred = (xt - torch.sqrt(alpha_bar) * x0) / torch.sqrt(1.0 - alpha_bar)
        alpha = self.alphas[t][:, None, None]
        beta = self.betas[t][:, None, None]
        mean = (xt - beta / torch.sqrt(1.0 - alpha_bar) * noise_pred) / torch.sqrt(alpha)
        nonzero = (t > 0).float()[:, None, None]
        variance = self.posterior_variance[t][:, None, None]
        return mean + nonzero * torch.sqrt(variance.clamp_min(1e-20)) * torch.randn_like(xt)

    def _physics_guided_output(
        self, xt, t, model_output, cond, wind_u, wind_v, previous_plan, scale
    ):
        """Apply smooth normalized guidance through the differentiable rollout."""
        if scale <= 0 or int(t[0]) >= config.PHYSICS_GUIDANCE_LAST_STEPS:
            return model_output
        with torch.enable_grad():
            x0 = self._x0_from_output(xt, t, model_output).detach().requires_grad_(True)
            states = rollout_controls_torch(x0, cond, wind_u, wind_v)
            cost, _ = candidate_cost_torch(
                states, x0, cond, previous_plan, wind_u, wind_v
            )
            gradient = torch.autograd.grad(cost.sum(), x0)[0]
            rms = gradient.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
            gradient = gradient / rms
        if self.prediction_type == "x0":
            return model_output - scale * gradient
        sigma = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return model_output + scale * sigma * gradient

    @staticmethod
    def _masked_mean(value, mask):
        expanded = mask[:, None].to(value.dtype)
        return (value * expanded).sum() / (expanded.sum() * value.shape[1]).clamp_min(1.0)

    def loss(
        self,
        controls,
        cond,
        wind_u,
        wind_v,
        target_states,
        valid_mask,
        previous_plan=None,
        previous_mask=None,
    ) -> Dict[str, torch.Tensor]:
        """Diffusion objective plus physical, progress and continuity constraints."""
        batch = controls.shape[0]
        device = controls.device
        t = torch.randint(0, self.num_steps, (batch,), device=device, dtype=torch.long)
        noise = torch.randn_like(controls)
        xt, noise = self.q_sample(controls, t, noise)
        wind_tokens = self.wind_encoder(wind_u, wind_v)

        train_cond = cond.clone()
        if self.cfg_dropout > 0:
            drop = torch.rand(batch, device=device) < self.cfg_dropout
            train_cond[drop] = self.unconditional_condition(train_cond[drop])

        train_previous = previous_plan
        if train_previous is None:
            train_previous = torch.zeros_like(controls)
        else:
            train_previous = train_previous.clone()
        if previous_mask is not None and config.PREVIOUS_PLAN_NOISE > 0:
            plan_noise = torch.randn_like(train_previous) * config.PREVIOUS_PLAN_NOISE
            train_previous = train_previous + plan_noise * previous_mask[:, None]
        if config.PREVIOUS_PLAN_DROPOUT > 0:
            drop_previous = torch.rand(batch, device=device) < config.PREVIOUS_PLAN_DROPOUT
            train_previous[drop_previous] = 0.0

        model_output = self.denoiser(xt, t, train_cond, wind_tokens, train_previous)
        diffusion_target = noise if self.prediction_type == "epsilon" else controls
        diffusion_loss = self._masked_mean((model_output - diffusion_target).square(), valid_mask)

        predicted_x0 = self._x0_from_output(xt, t, model_output).clamp(-1.5, 1.5)
        predicted_states = rollout_controls_torch(predicted_x0, cond, wind_u, wind_v)

        state_error = (predicted_states[:, :4] - target_states[:, :4]).square()
        energy_error = (predicted_states[:, 5:6] - target_states[:, 5:6]).square()
        state_loss = self._masked_mean(state_error, valid_mask) + self._masked_mean(
            energy_error, valid_mask
        )
        # Only the first point is executed before the next replan, so its
        # accuracy directly determines closed-loop behavior.
        first_step_loss = (
            predicted_states[:, :4, 0] - target_states[:, :4, 0]
        ).square().mean()

        valid_count = valid_mask.long().sum(dim=-1).clamp_min(1)
        last_idx = valid_count - 1
        batch_idx = torch.arange(batch, device=device)
        end_pos = predicted_states[batch_idx, :2, last_idx]
        start_pos = cond[:, [config.COND_X, config.COND_Y]]
        target = target_from_condition_torch(cond)
        start_distance = distance_km_torch(start_pos, target)
        end_distance = distance_km_torch(end_pos, target)
        expert_end_pos = target_states[batch_idx, :2, last_idx]
        expert_end_distance = distance_km_torch(expert_end_pos, target)
        expert_progress = F.relu(start_distance - expert_end_distance)
        predicted_progress = start_distance - end_distance
        expert_progress_loss = F.relu(
            config.EXPERT_PROGRESS_RATIO * expert_progress - predicted_progress
        ).mean() / max(config.WIDTH_KM, config.HEIGHT_KM)
        progress_loss = F.relu(
            end_distance - start_distance + config.PROGRESS_MARGIN_KM
        ).mean() / max(config.WIDTH_KM, config.HEIGHT_KM)

        x_km = predicted_states[:, 0] * config.WIDTH_KM
        y_km = predicted_states[:, 1] * config.HEIGHT_KM
        margin = config.BOUNDARY_MARGIN_KM
        boundary_error = (
            F.relu(margin - x_km)
            + F.relu(margin - (config.WIDTH_KM - x_km))
            + F.relu(margin - y_km)
            + F.relu(margin - (config.HEIGHT_KM - y_km))
        ) / margin
        boundary_error = boundary_error.masked_fill(~valid_mask, 0.0)
        boundary_loss = boundary_error.max(dim=-1).values.mean()
        energy_error = F.relu(config.ENERGY_SAFE_THRESHOLD - predicted_states[:, 5])
        energy_loss = (energy_error * valid_mask).sum() / valid_mask.sum().clamp_min(1)
        terminal_hour = predicted_states[batch_idx, 4, last_idx] * 24.0
        required_reserve = required_energy_reserve_torch(terminal_hour, end_distance)
        energy_loss = energy_loss + F.relu(
            required_reserve - predicted_states[batch_idx, 5, last_idx]
        ).mean()

        useful_progress = F.relu(start_distance - end_distance)
        energy_spent = F.relu(
            cond[:, config.COND_ENERGY] - predicted_states[batch_idx, 5, last_idx]
        )
        efficiency_loss = (
            energy_spent / (useful_progress / 100.0).clamp_min(0.05)
        ).clamp_max(5.0).mean()

        distance_series = distance_km_torch(
            predicted_states[:, :2].transpose(1, 2), target[:, None]
        )
        distance_series = torch.cat([start_distance[:, None], distance_series], dim=1)
        progress_pairs = valid_mask.clone()
        regression = F.relu(distance_series[:, 1:] - distance_series[:, :-1])
        progress_loss = progress_loss + (
            (regression * progress_pairs).sum() / progress_pairs.sum().clamp_min(1)
            / max(config.WIDTH_KM, config.HEIGHT_KM)
        )

        pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
        smooth_error = (predicted_x0[:, :, 1:] - predicted_x0[:, :, :-1]).square()
        smooth_loss = self._masked_mean(smooth_error, pair_mask)

        continuity_loss = torch.zeros((), device=device)
        if previous_plan is not None and previous_mask is not None and previous_mask.any():
            continuity_error = (predicted_x0 - previous_plan).square()
            continuity_loss = self._masked_mean(continuity_error, previous_mask)

        # Match the two failure-sensitive terms used by inference candidate
        # selection, instead of optimizing window reconstruction alone.
        _, physical = candidate_cost_torch(
            predicted_states, predicted_x0, cond, train_previous, wind_u, wind_v
        )
        closing_loss = physical["closing"].mean()
        stagnation_loss = physical["stagnation"].mean()

        total = (
            diffusion_loss
            + config.LOSS_STATE * state_loss
            + config.LOSS_FIRST_STEP * first_step_loss
            + config.LOSS_EXPERT_PROGRESS * expert_progress_loss
            + config.LOSS_PROGRESS * progress_loss
            + config.LOSS_BOUNDARY * boundary_loss
            + config.LOSS_ENERGY * energy_loss
            + config.LOSS_EFFICIENCY * efficiency_loss
            + config.LOSS_SMOOTH * smooth_loss
            + config.LOSS_CONTINUITY * continuity_loss
            + config.LOSS_CLOSING * closing_loss
            + config.LOSS_STAGNATION * stagnation_loss
        )
        return {
            "total": total,
            "diffusion": diffusion_loss.detach(),
            "state": state_loss.detach(),
            "first_step": first_step_loss.detach(),
            "expert_progress": expert_progress_loss.detach(),
            "progress": progress_loss.detach(),
            "boundary": boundary_loss.detach(),
            "energy": energy_loss.detach(),
            "efficiency": efficiency_loss.detach(),
            "smooth": smooth_loss.detach(),
            "continuity": continuity_loss.detach(),
            "closing": closing_loss.detach(),
            "stagnation": stagnation_loss.detach(),
        }

    @torch.no_grad()
    def sample_controls(
        self,
        cond,
        wind_u,
        wind_v,
        previous_plan=None,
        guidance_scale=2.0,
        sampler="ddpm",
        ddim_steps=50,
        eta=config.DDIM_ETA,
        physics_guidance_scale=config.PHYSICS_GUIDANCE_SCALE,
    ):
        batch = cond.shape[0]
        if previous_plan is None:
            previous_plan = torch.zeros(
                batch, self.action_dim, self.traj_len,
                device=cond.device, dtype=cond.dtype,
            )
        xt = torch.randn(batch, self.action_dim, self.traj_len, device=cond.device)
        wind_tokens = self.wind_encoder(wind_u, wind_v)

        if sampler == "ddpm":
            for step in reversed(range(self.num_steps)):
                t = torch.full((batch,), step, device=cond.device, dtype=torch.long)
                output = self._model_output(
                    xt, t, cond, wind_tokens, previous_plan, guidance_scale
                )
                output = self._physics_guided_output(
                    xt, t, output, cond, wind_u, wind_v, previous_plan,
                    physics_guidance_scale,
                )
                xt = self._ddpm_step(xt, t, output)
            return xt.clamp(-1.0, 1.0)

        if sampler != "ddim":
            raise ValueError(f"unsupported sampler: {sampler}")
        timesteps = np.linspace(0, self.num_steps - 1, ddim_steps, dtype=np.int64)[::-1]
        for index, step in enumerate(timesteps):
            t = torch.full((batch,), int(step), device=cond.device, dtype=torch.long)
            output = self._model_output(xt, t, cond, wind_tokens, previous_plan, guidance_scale)
            output = self._physics_guided_output(
                xt, t, output, cond, wind_u, wind_v, previous_plan,
                physics_guidance_scale,
            )
            x0 = self._x0_from_output(xt, t, output).clamp(-1.0, 1.0)
            alpha_bar = self.alphas_cumprod[t][:, None, None]
            epsilon = (xt - torch.sqrt(alpha_bar) * x0) / torch.sqrt(1.0 - alpha_bar)
            if index == len(timesteps) - 1:
                xt = x0
                continue
            previous_step = int(timesteps[index + 1])
            alpha_bar = self.alphas_cumprod[t][:, None, None]
            alpha_bar_prev = self.alphas_cumprod[previous_step]
            eta_value = torch.as_tensor(eta, device=xt.device, dtype=xt.dtype)
            if eta_value.ndim == 0:
                eta_value = eta_value.expand(batch)
            sigma = eta_value[:, None, None] * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar)
            ) * torch.sqrt(
                (1 - alpha_bar / alpha_bar_prev).clamp_min(0.0)
            )
            xt = (
                torch.sqrt(alpha_bar_prev) * x0
                + torch.sqrt((1 - alpha_bar_prev - sigma.square()).clamp_min(0.0)) * epsilon
                + sigma * torch.randn_like(xt)
            )
        return xt.clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample_multi(
        self,
        cond,
        wind_u,
        wind_v,
        previous_plan=None,
        guidance_scale=2.0,
        num_samples=5,
        sampler="ddpm",
        ddim_steps=50,
        eta=config.DDIM_ETA,
        physics_guidance_scale=config.PHYSICS_GUIDANCE_SCALE,
    ):
        """Generate candidates, physically roll them out and select minimum cost."""
        batch = cond.shape[0]
        cond_batch = cond.repeat_interleave(num_samples, dim=0)
        wind_u_batch = wind_u.repeat_interleave(num_samples, dim=0)
        wind_v_batch = wind_v.repeat_interleave(num_samples, dim=0)
        previous_batch = None
        if previous_plan is not None:
            previous_batch = previous_plan.repeat_interleave(num_samples, dim=0)

        # Pure Gaussian variation collapsed to nearly identical routes in v4.
        # Use training-supported previous-plan dropout/noise and several CFG /
        # DDIM profiles so candidates can leave a stale local route.
        candidate_id = torch.arange(
            batch * num_samples, device=cond.device
        ) % num_samples
        # Ask the same DiT for semantically different route families. This is
        # conditioning, not a second planner: every executed control remains
        # a diffusion-model output.
        route_mode = candidate_id % config.ROUTE_MODE_COUNT
        cond_batch[:, config.COND_ROUTE_MODE_START:config.COND_DIM] = 0.0
        cond_batch[
            torch.arange(batch * num_samples, device=cond.device),
            config.COND_ROUTE_MODE_START + route_mode,
        ] = 1.0
        if previous_batch is not None and num_samples > 1:
            cold = candidate_id % 4 == 1
            noisy = candidate_id % 4 == 2
            previous_batch[cold] = 0.0
            if noisy.any():
                previous_batch[noisy] = (
                    previous_batch[noisy]
                    + torch.randn_like(previous_batch[noisy])
                    * config.PREVIOUS_PLAN_CANDIDATE_NOISE
                ).clamp(-1.0, 1.0)
        profile = torch.tensor(
            [0.75, 1.0, 1.25, 0.60], device=cond.device, dtype=cond.dtype
        )
        guidance_batch = float(guidance_scale) * profile[candidate_id % len(profile)]
        eta_profile = torch.tensor(
            [0.75, 1.0, 1.5, 2.0], device=cond.device, dtype=cond.dtype
        )
        eta_batch = (float(eta) * eta_profile[candidate_id % len(eta_profile)]).clamp(0.0, 0.8)

        controls = self.sample_controls(
            cond_batch, wind_u_batch, wind_v_batch, previous_batch,
            guidance_batch, sampler, ddim_steps, eta_batch,
            physics_guidance_scale,
        )
        states = rollout_controls_torch(controls, cond_batch, wind_u_batch, wind_v_batch)
        costs, components = candidate_cost_torch(
            states, controls, cond_batch, previous_batch,
            wind_u_batch, wind_v_batch,
        )

        margin_x = config.BOUNDARY_MARGIN_KM / config.WIDTH_KM
        margin_y = config.BOUNDARY_MARGIN_KM / config.HEIGHT_KM
        feasible = (
            torch.isfinite(states).all(dim=(1, 2))
            & torch.isfinite(controls).all(dim=(1, 2))
            & (states[:, 0] >= margin_x).all(dim=1)
            & (states[:, 0] <= 1.0 - margin_x).all(dim=1)
            & (states[:, 1] >= margin_y).all(dim=1)
            & (states[:, 1] <= 1.0 - margin_y).all(dim=1)
            & (states[:, 5] >= config.ENERGY_SAFE_THRESHOLD).all(dim=1)
        )
        costs = costs + (~feasible).to(costs.dtype) * 1.0e6

        controls = controls.view(batch, num_samples, self.action_dim, self.traj_len)
        states = states.view(batch, num_samples, config.STATE_DIM, self.traj_len)
        costs = costs.view(batch, num_samples)
        feasible = feasible.view(batch, num_samples)
        best_index = costs.argmin(dim=1)
        row = torch.arange(batch, device=cond.device)
        return {
            "controls": controls,
            "states": states,
            "costs": costs,
            "best_index": best_index,
            "route_mode": route_mode.view(batch, num_samples),
            "best_route_mode": route_mode.view(batch, num_samples)[row, best_index],
            "best_controls": controls[row, best_index],
            "best_states": states[row, best_index],
            "feasible": feasible,
            "components": components,
        }
