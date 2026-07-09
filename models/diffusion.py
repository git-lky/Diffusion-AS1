"""DDPM/DDIM sampler for trajectory generation."""
import torch
import torch.nn as nn
import numpy as np
import config

class TrajectoryDiffusion(nn.Module):
    def __init__(self, unet, wind_encoder, cfg_dropout=0.1):
        super().__init__()
        self.unet = unet
        self.wind_encoder = wind_encoder
        betas = config.cosine_beta_schedule(config.DIFFUSION_STEPS)
        betas = torch.tensor(betas, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

        self.num_steps = config.DIFFUSION_STEPS
        self.cfg_dropout = cfg_dropout
        self.state_dim = config.STATE_DIM
        self.traj_len = config.TRAJ_LEN

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alphas_cumprod[t][:, None, None]
        b = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return a * x0 + b * noise, noise

    def p_sample(self, xt, t, cond, wind_u, wind_v):
        wind_enc = self.wind_encoder(wind_u, wind_v)
        noise_pred = self.unet(xt, t, cond, wind_enc)
        alpha = self.alphas[t][:, None, None]
        alpha_cumprod = self.alphas_cumprod[t][:, None, None]
        beta = self.betas[t][:, None, None]
        coef = beta / torch.sqrt(1.0 - alpha_cumprod)
        mean = (1.0 / torch.sqrt(alpha)) * (xt - coef * noise_pred)
        if t[0] > 0:
            noise = torch.randn_like(xt)
            var = self.posterior_variance[t][:, None, None]
            return mean + torch.sqrt(var) * noise
        return mean

    @torch.no_grad()

    @torch.no_grad()
    def sample(self, cond, wind_u, wind_v):
        """Generate trajectory from noise given conditions (256 steps)."""
        B = cond.shape[0]
        device = cond.device
        xt = torch.randn(B, self.state_dim, self.traj_len, device=device)
        for step in reversed(range(self.num_steps)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            xt = self.p_sample(xt, t, cond, wind_u, wind_v)
        return xt

    @torch.no_grad()
    def sample_with_guidance(self, cond, wind_u, wind_v, guidance_scale=2.0):
        """CFG sampling: push trajectory toward the given condition."""
        B = cond.shape[0]
        device = cond.device
        xt = torch.randn(B, self.state_dim, self.traj_len, device=device)
        zero_cond = torch.zeros_like(cond)
        for step in reversed(range(self.num_steps)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            wind_enc = self.wind_encoder(wind_u, wind_v)
            noise_pred_cond = self.unet(xt, t, cond, wind_enc)
            noise_pred_uncond = self.unet(xt, t, zero_cond, wind_enc)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            alpha = self.alphas[t][:, None, None]
            alpha_cumprod = self.alphas_cumprod[t][:, None, None]
            beta = self.betas[t][:, None, None]
            coef = beta / torch.sqrt(1.0 - alpha_cumprod)
            mean = (1.0 / torch.sqrt(alpha)) * (xt - coef * noise_pred)
            if step > 0:
                noise = torch.randn_like(xt)
                var = self.posterior_variance[t][:, None, None]
                xt = mean + torch.sqrt(var) * noise
            else:
                xt = mean
        return xt


    def sample_ddim(self, cond, wind_u, wind_v, ddim_steps=20):
        B = cond.shape[0]
        device = cond.device
        stride = max(1, self.num_steps // ddim_steps)
        xt = torch.randn(B, self.state_dim, self.traj_len, device=device)
        for step in reversed(range(0, self.num_steps, stride)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            xt = self.p_sample(xt, t, cond, wind_u, wind_v)
        return xt


    @torch.no_grad()
    def sample_with_multi_guidance(self, cond, wind_u, wind_v, guidance_scale=2.0,
                                    progress_w=5.0, boundary_w=10.0, energy_w=3.0):
        """CFG + classifier guidance on progress, boundary, energy."""
        B = cond.shape[0]
        device = cond.device
        xt = torch.randn(B, self.state_dim, self.traj_len, device=device)
        zero_cond = torch.zeros_like(cond)
        for step in reversed(range(self.num_steps)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            wind_enc = self.wind_encoder(wind_u, wind_v)
            # CFG
            noise_pred_cond = self.unet(xt, t, cond, wind_enc)
            noise_pred_uncond = self.unet(xt, t, zero_cond, wind_enc)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            # Classifier guidance on later steps
            if step < 32:
                with torch.enable_grad():
                    alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1)
                    sigma = torch.sqrt(1.0 - alpha_bar)
                    x0 = ((xt - sigma * noise_pred) / torch.sqrt(alpha_bar)).detach().requires_grad_(True)
                    losses = []
                    if progress_w > 0:
                        gx = cond[:, 6:7].unsqueeze(-1); gy = cond[:, 7:8].unsqueeze(-1)
                        d = torch.sqrt((x0[:, 0:1] - gx)**2 + (x0[:, 1:2] - gy)**2)
                        losses.append(progress_w * torch.clamp(d[:, :, 1:] - d[:, :, :-1], min=0).mean())
                    if boundary_w > 0:
                        xs, ys = x0[:, 0], x0[:, 1]
                        losses.append(boundary_w * (torch.clamp(-xs, min=0) + torch.clamp(xs-1, min=0)
                                                    + torch.clamp(-ys, min=0) + torch.clamp(ys-1, min=0)).mean())
                    if energy_w > 0:
                        losses.append(energy_w * (x0[:, 2]**2 + x0[:, 3]**2).mean())
                    if losses:
                        energy = sum(losses)
                        grad = torch.autograd.grad(energy, x0)[0]
                        noise_pred = noise_pred + sigma / torch.sqrt(alpha_bar.clamp(min=1e-6)) * grad
            # DDPM step
            alpha = self.alphas[t][:, None, None]
            alpha_cumprod = self.alphas_cumprod[t][:, None, None]
            beta = self.betas[t][:, None, None]
            coef = beta / torch.sqrt(1.0 - alpha_cumprod)
            mean = (1.0 / torch.sqrt(alpha)) * (xt - coef * noise_pred)
            if step > 0:
                noise = torch.randn_like(xt)
                var = self.posterior_variance[t][:, None, None]
                xt = mean + torch.sqrt(var) * noise
            else:
                xt = mean
        return xt

    @torch.no_grad()
    def sample_multi_pick_best(self, cond, wind_u, wind_v, guidance_scale=2.0, num_samples=20):
        """Sample multiple trajectories and pick the one closest to target."""
        B = cond.shape[0]
        device = cond.device
        cond_batch = cond.repeat_interleave(num_samples, dim=0)
        wind_u_batch = wind_u.repeat_interleave(num_samples, dim=0)
        wind_v_batch = wind_v.repeat_interleave(num_samples, dim=0)
        trajs = self.sample_with_guidance(cond_batch, wind_u_batch, wind_v_batch, guidance_scale)
        trajs = trajs.view(B, num_samples, self.state_dim, self.traj_len)
        tx = cond[:, 6:7].unsqueeze(-1)
        ty = cond[:, 7:8].unsqueeze(-1)
        end_x = trajs[:, :, 0:1, -1]
        end_y = trajs[:, :, 1:2, -1]
        dist = torch.sqrt((end_x - tx)**2 + (end_y - ty)**2).squeeze(-1).squeeze(-1)
        best_idx = dist.argmin(dim=1)
        best = trajs[torch.arange(B, device=device), best_idx]
        return best

    def loss(self, x0, cond, wind_u, wind_v):
        B = x0.shape[0]
        device = x0.device
        t = torch.randint(0, self.num_steps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt, noise = self.q_sample(x0, t, noise)
        wind_enc = self.wind_encoder(wind_u, wind_v)
        # CFG: randomly drop condition for 10% of samples
        if self.cfg_dropout > 0:
            mask = torch.rand(B, device=device) < self.cfg_dropout
            cond = cond.clone()
            cond[mask] = 0
        noise_pred = self.unet(xt, t, cond, wind_enc)
        return nn.functional.mse_loss(noise_pred, noise)
