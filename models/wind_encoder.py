"""Spatiotemporal encoder for dynamic wind context."""
import torch
import torch.nn as nn

import config


class SpatioTemporalWindEncoder(nn.Module):
    """Return spatial wind tokens for every observed wind instant.

    Input shape is ``(B,T,H,W)`` for each wind component.  The result has
    shape ``(B,F,P,D)`` where P is a small spatial patch grid.  No temporal or
    spatial mean pooling is used as the sole conditioning signal.
    """

    def __init__(self, out_dim=config.WIND_ENC_OUT, num_frames=config.WIND_NUM_FRAMES):
        super().__init__()
        self.out_dim = out_dim
        self.num_frames = num_frames
        self.patch_y = config.WIND_PATCHES_Y
        self.patch_x = config.WIND_PATCHES_X
        self.num_patches = self.patch_y * self.patch_x

        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, config.WIND_PATCH_CHANNELS, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((self.patch_y, self.patch_x)),
        )
        self.patch_proj = nn.Linear(config.WIND_PATCH_CHANNELS, out_dim)
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, self.num_patches, out_dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, num_frames, 1, out_dim))

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=4,
            dim_feedforward=out_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=2)
        self.out_norm = nn.LayerNorm(out_dim)

        nn.init.normal_(self.spatial_pos, std=0.02)
        nn.init.normal_(self.temporal_pos, std=0.02)

    def forward(self, wind_u_frames, wind_v_frames):
        if wind_u_frames.shape != wind_v_frames.shape:
            raise ValueError("wind_u and wind_v must have identical shapes")
        batch, frames, height, width = wind_u_frames.shape
        if frames > self.num_frames:
            raise ValueError(f"received {frames} wind frames, maximum is {self.num_frames}")

        scale = float(config.WIND_MAX_SPEED)
        wind = torch.stack([wind_u_frames, wind_v_frames], dim=2) / scale
        wind = wind.reshape(batch * frames, 2, height, width)
        patches = self.conv(wind)
        patches = patches.flatten(2).transpose(1, 2)
        patches = self.patch_proj(patches)
        patches = patches.view(batch, frames, self.num_patches, self.out_dim)
        patches = patches + self.spatial_pos + self.temporal_pos[:, :frames]

        # Temporal context is computed from each frame's global summary and then
        # added back to all spatial patches at that same instant.
        temporal = patches.mean(dim=2)
        temporal = self.temporal_encoder(temporal)
        patches = patches + temporal.unsqueeze(2)
        return self.out_norm(patches)


# Compatibility names used by older entry points.  Both now preserve the full
# time/space token grid instead of returning one pooled vector.
TemporalWindEncoder = SpatioTemporalWindEncoder
WindEncoder = SpatioTemporalWindEncoder
