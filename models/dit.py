"""DiT: Diffusion Transformer for trajectory planning."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config

#正弦位置编码
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)  #递减因子
        emb = torch.exp(torch.arange(half, device=device).float() * -emb)
        emb = t.unsqueeze(-1).float() * emb.view(*([1] * t.ndim), half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

#自适应层归一化 Transformer 块
class AdaLNZeroBlock(nn.Module):
    """Transformer block with adaptive layer norm + zero-init."""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
         # MLP: 256 → 1024 → 256
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model), nn.Dropout(dropout)
        )
         # adaLN: 256 → 1536 (1536 = 6 × 256，生成 6 组参数)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        # zero init output layers
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)

    def forward(self, x, c_emb):
        # x: (B, N, D), c_emb: (B, D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN(c_emb).chunk(6, dim=-1)
        # self-attention
        x_norm = self.norm1(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa[:, None] * attn_out
        # mlp
        x_norm = self.norm2(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        x = x + gate_mlp[:, None] * self.mlp(x_norm)
        return x

class TrajectoryDiT(nn.Module):
    """Diffusion Transformer for trajectory denoising."""
    def __init__(self):
        super().__init__()
        self.state_dim = config.STATE_DIM
        self.traj_len = config.TRAJ_LEN
        self.d_model = 256
        self.n_layers = 6
        self.nhead = 8
        self.cond_in_dim = 8   # start_state(6) + goal(2)
        self.wind_dim = config.WIND_ENC_OUT

        # Input projections
        self.traj_proj = nn.Linear(self.state_dim, self.d_model)     # Linear(6, 256)  → 把轨迹每步 6 个值映射到 256 维
        self.cond_proj = nn.Linear(self.cond_in_dim, self.d_model)      # Linear(8, 256)  → 把条件向量映射到 256 维
        self.wind_proj = nn.Linear(self.wind_dim, self.d_model)     # Linear(64, 256) → 把风场编码映射到 256 维
        self.goal_proj = nn.Linear(2, self.d_model)   # Linear(2, 256)  → 把目标位置 [tx, ty] 映射到 256 维

        # Positional encoding
        self.pos_emb = SinusoidalPosEmb(self.d_model)

        # DiT blocks
        self.blocks = nn.ModuleList([
            AdaLNZeroBlock(self.d_model, self.nhead)
            for _ in range(self.n_layers)
        ])

        # Condition embedding
        self.cond_emb_proj = nn.Sequential(
            nn.Linear(self.d_model + self.d_model * 2, self.d_model),
            nn.SiLU(), nn.Linear(self.d_model, self.d_model)
        )

        # Output projection
        self.norm_out = nn.LayerNorm(self.d_model, elementwise_affine=False)
        self.proj_out = nn.Linear(self.d_model, self.state_dim)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def forward(self, x_t, t, cond, wind_enc):
        """x_t: (B, state_dim, T), t: (B,), cond: (B, 8), wind_enc: (B, wind_dim)"""
        B, D, T = x_t.shape
        device = x_t.device

        # Tokenize trajectory: (B, T, D)
        x = x_t.permute(0, 2, 1)
        x = self.traj_proj(x)

        # Goal broadcasting: add target position to every trajectory token
        goal = cond[:, 6:8]                         # (B, 2) target position
        goal_emb = self.goal_proj(goal).unsqueeze(1) # (B, 1, d_model)
        x = x + goal_emb                             # broadcast to all T steps

        # Positional encoding
        pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1) / T
        x = x + self.pos_emb(pos)

        # Condition tokens
        cond_emb = self.cond_proj(cond)  # (B, d_model)
        wind_emb = self.wind_proj(wind_enc)  # (B, d_model)
        # concat as two condition tokens: start+goal | wind
        c_tokens = torch.stack([
            cond_emb,
            wind_emb
        ], dim=1)  # (B, 2, d_model)

        # All tokens
        tokens = torch.cat([x, c_tokens], dim=1)  # (B, T+2, d_model)

        # Time embedding
        t_emb = self.pos_emb(t)  # (B, d_model)
        # Condition embedding = concat(t_emb + pool(cond_emb) + pool(wind_emb))
        c_full = torch.cat([t_emb, cond_emb, wind_emb], dim=-1)
        c_emb = self.cond_emb_proj(c_full)

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, c_emb)

        # Extract trajectory tokens and project
        x_out = tokens[:, :T]  # (B, T, d_model)
        x_out = self.norm_out(x_out)
        x_out = self.proj_out(x_out)  # (B, T, state_dim)
        return x_out.permute(0, 2, 1)  # (B, state_dim, T)
