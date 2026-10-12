"""Paper-style static and animated rolling trajectory visualizations."""
from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffusion_as1_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/diffusion_as1_cache")
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

import config


def _wind_coordinates(shape):
    height, width = shape
    x = np.linspace(0, config.WIDTH, width)
    y = np.linspace(0, config.HEIGHT, height)
    return np.meshgrid(x, y)


def _format_clock(hours):
    total_minutes = int(round(float(hours) * 60.0)) % (24 * 60)
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def draw_paper_style(ax, frame, start_pos, target_pos, wind_limits, show_candidates=False):
    """Draw one subplot in the visual style used by paper Figures 4 and 6."""
    wind_u = frame["wind_u"]
    wind_v = frame["wind_v"]
    speed = np.sqrt(wind_u ** 2 + wind_v ** 2)
    x, y = _wind_coordinates(speed.shape)
    vmin, vmax = wind_limits
    levels = np.linspace(vmin, vmax, 8)
    contour = ax.contourf(x, y, speed, levels=levels, cmap="viridis", extend="max", zorder=0)
    ax.quiver(
        x, y, wind_u, wind_v, color="black", angles="xy", scale_units="xy",
        scale=0.65, width=0.0022, headwidth=3.0, alpha=0.9, zorder=1,
    )

    if show_candidates:
        for candidate in frame.get("candidates", []):
            if candidate.shape[-1] == 0:
                continue
            ax.plot(
                candidate[0] * config.WIDTH, candidate[1] * config.HEIGHT,
                color="lightgray", linewidth=0.7, alpha=0.35, zorder=2,
            )

    history = frame["history"]
    history_line, = ax.plot(
        history[0] * config.WIDTH, history[1] * config.HEIGHT,
        color="white", linewidth=2.5, zorder=4, label="Executed trajectory",
    )
    history_line.set_path_effects([
        path_effects.Stroke(linewidth=3.5, foreground="black", alpha=0.35),
        path_effects.Normal(),
    ])

    prediction = frame.get("prediction")
    if prediction is not None and prediction.shape[-1] > 0:
        prediction_x = np.concatenate([[frame["current"][0]], prediction[0]]) * config.WIDTH
        prediction_y = np.concatenate([[frame["current"][1]], prediction[1]]) * config.HEIGHT
        ax.plot(
            prediction_x, prediction_y,
            color="red", linestyle="--", linewidth=1.8, zorder=5,
            label="Current 4 h DiT prediction",
        )

    current = frame["current"]
    ax.scatter(
        [start_pos[0]], [start_pos[1]], s=55, facecolor="white", edgecolor="black",
        linewidth=1.0, zorder=7, label="Start",
    )
    ax.scatter(
        [target_pos[0]], [target_pos[1]], s=120, facecolor="red", edgecolor="red",
        zorder=7, label="Target",
    )
    ax.scatter(
        [current[0] * config.WIDTH], [current[1] * config.HEIGHT], s=65,
        facecolor="dodgerblue", edgecolor="white", linewidth=1.0,
        zorder=8, label="Current position",
    )

    radius_x = config.TARGET_RANGE / config.WIDTH_KM * config.WIDTH
    ax.add_patch(plt.Circle(
        target_pos, radius_x, fill=False, color="red", linestyle=":",
        linewidth=1.0, alpha=0.8, zorder=6,
    ))
    ax.set_xlim(0, config.WIDTH)
    ax.set_ylim(0, config.HEIGHT)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    departure = _format_clock(frame.get("start_time_hours", 0.0))
    local_time = _format_clock(
        frame.get("clock_time_hours", frame.get("time_hours", 0.0))
    )
    ax.set_title(
        f"departure={departure} | local={local_time} | elapsed={frame['time_hours']:.1f} h\n"
        f"energy={frame['energy'] * 100:.0f}% | "
        f"distance={frame['distance_km']:.1f} km"
    )
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    return contour


def _render_frame(frame, start_pos, target_pos, wind_limits, dpi=120):
    # 960x576 at 120 dpi: dimensions are divisible by 16 for H.264 encoding.
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=dpi)
    contour = draw_paper_style(ax, frame, start_pos, target_pos, wind_limits)
    colorbar = fig.colorbar(contour, ax=ax, pad=0.02)
    colorbar.set_label("Wind speed (m/s)")
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    image = rgba[..., :3].copy()
    plt.close(fig)
    return image


def _wind_limits(frames):
    maximum = max(
        float(np.sqrt(frame["wind_u"] ** 2 + frame["wind_v"] ** 2).max())
        for frame in frames
    )
    return 0.0, max(1.0, float(np.ceil(maximum)))


def save_static_result(result, path, frame=None):
    frame = frame or result.frames[-1]
    limits = _wind_limits(result.frames)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    contour = draw_paper_style(ax, frame, result.start_pos, result.target_pos, limits)
    colorbar = fig.colorbar(contour, ax=ax, pad=0.02)
    colorbar.set_label("Wind speed (m/s)")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_animation(
    result,
    gif_path,
    gif_fps=12.0,
    frames_dir=None,
):
    limits = _wind_limits(result.frames)
    images = [
        _render_frame(frame, result.start_pos, result.target_pos, limits)
        for frame in result.frames
    ]
    if frames_dir is not None:
        os.makedirs(frames_dir, exist_ok=True)
        for index, image in enumerate(images):
            imageio.imwrite(os.path.join(frames_dir, f"frame_{index:04d}.png"), image)
    # Pillow's GIF writer expects milliseconds.  Passing seconds directly is
    # rounded to a near-zero delay and makes the animation appear too fast.
    gif_duration_ms = max(10, int(round(1000.0 / gif_fps)))
    imageio.mimsave(gif_path, images, duration=gif_duration_ms, loop=0)


def save_diagnostic_plot(result, path):
    states = result.states
    time_hours = np.arange(states.shape[1]) * config.DT / 3600.0
    target = np.array([
        result.target_pos[0] / config.WIDTH,
        result.target_pos[1] / config.HEIGHT,
    ])
    dx = (states[0] - target[0]) * config.WIDTH_KM
    dy = (states[1] - target[1]) * config.HEIGHT_KM
    distance = np.sqrt(dx * dx + dy * dy)
    speed = np.linalg.norm(states[2:4], axis=0) * config.MAX_SPEED

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(time_hours, distance, color="tab:red")
    axes[0].set_ylabel("Distance (km)")
    axes[1].plot(time_hours, states[5] * 100, color="tab:green")
    axes[1].axhline(config.ENERGY_SAFE_THRESHOLD * 100, color="red", linestyle="--")
    axes[1].set_ylabel("Energy (%)")
    axes[2].plot(time_hours, speed, color="tab:blue")
    axes[2].axhline(config.MAX_SPEED, color="red", linestyle="--")
    axes[2].set_ylabel("Speed (m/s)")
    axes[2].set_xlabel("Mission time (h)")
    for axis in axes:
        axis.grid(alpha=0.3)
        for replan in np.arange(0, time_hours[-1] + 1e-9, config.REPLAN_MINUTES / 60):
            axis.axvline(replan, color="gray", linewidth=0.35, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_result_visualizations(
    result,
    output_dir,
    stem="rolling_trajectory",
    gif_fps=12.0,
):
    os.makedirs(output_dir, exist_ok=True)
    png = os.path.join(output_dir, f"{stem}.png")
    gif = os.path.join(output_dir, f"{stem}.gif")
    diagnostic = os.path.join(output_dir, f"{stem}_diagnostics.png")
    frames_dir = os.path.join(output_dir, f"{stem}_frames")
    save_static_result(result, png)
    save_animation(result, gif, gif_fps=gif_fps, frames_dir=frames_dir)
    save_diagnostic_plot(result, diagnostic)
    return {
        "png": png, "gif": gif,
        "diagnostic": diagnostic,
        "frames": frames_dir,
    }
