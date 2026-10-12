"""Model factory."""
from .dit import TrajectoryDiT
import config

def create_denoiser():
    if config.BACKBONE == "dit":
        return TrajectoryDiT()
    raise ValueError(
        f"Rolling-horizon velocity-delta generation currently requires the DiT backbone; "
        f"got {config.BACKBONE!r}"
    )
