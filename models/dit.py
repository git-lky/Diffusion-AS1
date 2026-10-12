"""Rolling-horizon Diffusion Transformer for airship trajectory generation."""
import math

import torch
import torch.nn as nn

import config


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, value):
        half = self.dim // 2
        factor = math.log(10000) / max(half - 1, 1)
        frequencies = torch.exp(
            torch.arange(half, device=value.device, dtype=torch.float32) * -factor
        )
        phase = value.float().unsqueeze(-1) * frequencies
        return torch.cat([phase.sin(), phase.cos()], dim=-1)


class AdaLNWindBlock(nn.Module):
    """adaLN-Zero block with time-aligned spatial wind cross-attention."""

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.wind_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_self = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm_wind = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm_mlp = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 9 * d_model))
        nn.init.constant_(self.ada_ln[-1].weight, 0)
        nn.init.constant_(self.ada_ln[-1].bias, 0)

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1 + scale[:, None]) + shift[:, None]

    def forward(self, x, condition, wind_tokens):
        params = self.ada_ln(condition).chunk(9, dim=-1)
        shift_s, scale_s, gate_s, shift_w, scale_w, gate_w, shift_m, scale_m, gate_m = params

        x_norm = self._modulate(self.norm_self(x), shift_s, scale_s)
        self_out, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + gate_s[:, None] * self_out

        # Every one-hour control token can inspect the complete two-hour wind
        # context. This preserves a single DiT planner while exposing changes
        # that occur beyond the generated control horizon.
        batch, _, dim = x.shape
        query = self._modulate(self.norm_wind(x), shift_w, scale_w)
        keys = wind_tokens.reshape(batch, -1, dim)
        wind_out, _ = self.wind_attn(query, keys, keys, need_weights=False)
        x = x + gate_w[:, None] * wind_out

        x_norm = self._modulate(self.norm_mlp(x), shift_m, scale_m)
        x = x + gate_m[:, None] * self.mlp(x_norm)
        return x


class TrajectoryDiT(nn.Module):
    """Denoise a one-hour normalized airspeed-setpoint sequence."""

    def __init__(self):
        super().__init__()
        self.action_dim = config.ACTION_DIM
        self.traj_len = config.TRAJ_LEN
        self.d_model = config.D_MODEL

        self.action_proj = nn.Linear(self.action_dim, self.d_model)
        # No bias: an all-zero cold-start plan must be exactly neutral and
        # behave identically whether it is explicitly supplied or synthesized.
        self.previous_proj = nn.Linear(self.action_dim, self.d_model, bias=False)
        self.condition_proj = nn.Linear(config.COND_DIM, self.d_model)
        self.goal_proj = nn.Linear(2, self.d_model)
        self.wind_proj = nn.Linear(config.WIND_ENC_OUT, self.d_model)
        self.diffusion_time_emb = SinusoidalPosEmb(self.d_model)
        self.trajectory_pos_emb = SinusoidalPosEmb(self.d_model)

        self.condition_fusion = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model), nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.blocks = nn.ModuleList([
            AdaLNWindBlock(self.d_model, config.DIT_HEADS) for _ in range(config.DIT_LAYERS)
        ])
        self.norm_out = nn.LayerNorm(self.d_model, elementwise_affine=False)
        self.proj_out = nn.Linear(self.d_model, self.action_dim)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def forward(self, x_t, diffusion_t, cond, wind_tokens, previous_plan=None):
        """
        Args:
            x_t: ``(B,2,T)`` noisy normalized airspeed setpoints.
            diffusion_t: ``(B,)`` diffusion indices.
            cond: ``(B,10)`` rolling condition.
            wind_tokens: ``(B,F,P,wind_dim)`` wind-context tokens; F may exceed T.
            previous_plan: optional shifted previous plan, ``(B,2,T)``.
        """
        batch, _, steps = x_t.shape
        if wind_tokens.shape[1] < steps:
            raise ValueError(
                f"wind context is too short: {wind_tokens.shape[1]} frames for "
                f"{steps} trajectory steps"
            )

        x = self.action_proj(x_t.transpose(1, 2))
        if previous_plan is not None:
            x = x + self.previous_proj(previous_plan.transpose(1, 2))

        pos = torch.arange(steps, device=x.device, dtype=torch.float32)
        pos = pos.unsqueeze(0).expand(batch, -1) / max(steps - 1, 1)
        x = x + self.trajectory_pos_emb(pos)

        relative_goal = cond[:, [config.COND_GOAL_DX, config.COND_GOAL_DY]]
        x = x + self.goal_proj(relative_goal).unsqueeze(1)

        wind_tokens = self.wind_proj(wind_tokens)
        cond_emb = self.condition_proj(cond)
        wind_global = wind_tokens.mean(dim=(1, 2))
        time_emb = self.diffusion_time_emb(diffusion_t)
        global_condition = self.condition_fusion(torch.cat([time_emb, cond_emb, wind_global], dim=-1))

        for block in self.blocks:
            x = block(x, global_condition, wind_tokens)

        return self.proj_out(self.norm_out(x)).transpose(1, 2)
