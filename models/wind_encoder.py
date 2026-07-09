"""Multi-frame CNN encoder for wind field -> latent vector."""
import torch
import torch.nn as nn
import config

class WindEncoder(nn.Module):
    """Encode 4 wind field frames (steps 0, 16, 32, 48) into latent vector."""
    def __init__(self, out_dim=config.WIND_ENC_OUT, num_frames=4):
        super().__init__()
        H, W = config.HEIGHT // config.RANGE_SCALE, config.WIDTH // config.RANGE_SCALE
        self.num_frames = num_frames

        # Shared CNN for each frame
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 2, H, W)
            conv_out = self.conv(dummy)
            flat_dim = conv_out.numel() // conv_out.shape[0]
        # Fusion: flatten each frame, then merge all frames
        self.fc = nn.Linear(flat_dim * num_frames, out_dim)

    def forward(self, wind_u_frames, wind_v_frames):
        """wind_u_frames: (B, num_frames, H, W); wind_v_frames: same shape"""
        B, F, H, W = wind_u_frames.shape
        feats = []
        for f in range(F):
            u = wind_u_frames[:, f]  # (B, H, W)
            v = wind_v_frames[:, f]
            x = torch.stack([u, v], dim=1)  # (B, 2, H, W)
            x = self.conv(x)
            feats.append(x.view(B, -1))
        return self.fc(torch.cat(feats, dim=-1))


class TemporalWindEncoder(nn.Module):
    """Encode multiple wind field frames with temporal attention across time steps."""
    def __init__(self, out_dim=config.WIND_ENC_OUT, num_frames=8):
        super().__init__()
        H, W = config.HEIGHT // config.RANGE_SCALE, config.WIDTH // config.RANGE_SCALE
        self.num_frames = num_frames

        # Shared CNN backbone for each frame (same structure as WindEncoder)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 2, H, W)
            conv_out = self.conv(dummy)
            self.flat_dim = conv_out.numel() // conv_out.shape[0]

        # Temporal attention across frames
        self.temporal_attn = nn.MultiheadAttention(self.flat_dim, num_heads=4, batch_first=True)
        # Final projection: mean-pooled features -> out_dim
        self.fc = nn.Linear(self.flat_dim, out_dim)

    def forward(self, wind_u_frames, wind_v_frames):
        """wind_u_frames: (B, num_frames, H, W); wind_v_frames: same shape"""
        B, F, H, W = wind_u_frames.shape
        feats = []
        for f in range(F):
            u = wind_u_frames[:, f]   # (B, H, W)
            v = wind_v_frames[:, f]
            x = torch.stack([u, v], dim=1)   # (B, 2, H, W)
            x = self.conv(x)
            feats.append(x.view(B, -1))      # (B, flat_dim)
        feats = torch.stack(feats, dim=1)    # (B, F, flat_dim)
        # Temporal self-attention: let frames exchange information
        attn_out, _ = self.temporal_attn(feats, feats, feats)   # (B, F, flat_dim)
        # Mean pool across the time dimension
        pooled = attn_out.mean(dim=1)        # (B, flat_dim)
        return self.fc(pooled)