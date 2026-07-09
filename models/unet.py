"""1D U-Net for trajectory denoising with conditioning."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config

def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half).float() / half).to(t.device)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, cond_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.cond_mlp = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, out_ch))
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, cond):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_mlp(t_emb)[:, :, None] + self.cond_mlp(cond)[:, :, None]
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.shortcut(x)

class TrajectoryUNet(nn.Module):
    """1D U-Net that denoises trajectories of shape (B, STATE_DIM, TRAJ_LEN)."""
    def __init__(self):
        super().__init__()
        self.state_dim = config.STATE_DIM
        self.traj_len = config.TRAJ_LEN
        self.dim = config.DIM
        dims = [self.dim * m for m in config.DIM_MULTS]
        time_dim = self.dim * 4
        cond_dim = config.CONDITION_DIM

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.input_proj = nn.Conv1d(self.state_dim, self.dim, 1)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        in_ch = self.dim
        for out_ch in dims:
            self.down_blocks.append(nn.ModuleList([
                ResBlock1D(in_ch, out_ch, time_dim, cond_dim),
                ResBlock1D(out_ch, out_ch, time_dim, cond_dim),
            ]))
            self.downsample.append(nn.Conv1d(out_ch, out_ch, 3, stride=2, padding=1))
            in_ch = out_ch

        self.mid_block = nn.ModuleList([
            ResBlock1D(in_ch, in_ch, time_dim, cond_dim),
            ResBlock1D(in_ch, in_ch, time_dim, cond_dim),
        ])

        for out_ch in reversed(dims):
            self.up_blocks.append(nn.ModuleList([
                ResBlock1D(in_ch + out_ch, out_ch, time_dim, cond_dim),
                ResBlock1D(out_ch, out_ch, time_dim, cond_dim),
            ]))
            in_ch = out_ch

        self.output = nn.Sequential(nn.GroupNorm(8, in_ch), nn.SiLU(), nn.Conv1d(in_ch, self.state_dim, 1))

    def forward(self, x, t, cond, wind_enc):
        """x: (B, STATE_DIM, T), t: (B,), cond: (B, 8), wind_enc: (B, WIND_ENC_OUT)"""
        full_cond = torch.cat([cond, wind_enc], dim=-1)
        t_emb = timestep_embedding(t, self.dim * 4)
        t_emb = self.time_mlp(t_emb)

        h = self.input_proj(x)
        hs = []
        for (b1, b2), ds in zip(self.down_blocks, self.downsample):
            h = b1(h, t_emb, full_cond)
            h = b2(h, t_emb, full_cond)
            hs.append(h)
            h = ds(h)

        for b1, b2 in [self.mid_block]:
            h = b1(h, t_emb, full_cond)
            h = b2(h, t_emb, full_cond)

        for (b1, b2), skip in zip(self.up_blocks, reversed(hs)):
            h = F.interpolate(h, size=skip.shape[-1], mode='nearest')
            h = torch.cat([h, skip], dim=1)
            h = b1(h, t_emb, full_cond)
            h = b2(h, t_emb, full_cond)

        return self.output(h)
