"""Model factory."""
from .dit import TrajectoryDiT
from .unet import TrajectoryUNet
import config

def create_denoiser():
    if config.BACKBONE == "dit":
        return TrajectoryDiT()
    elif config.BACKBONE == "unet":
        return TrajectoryUNet()
    raise ValueError(f"Unknown backbone: {config.BACKBONE}")
