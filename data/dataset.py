"""Generate training trajectories using airship dynamics."""
import os, sys, math, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from dynamics.airship import ASUnit_low
from dynamics.wind_field import WindGenerator
from dynamics.memory import Memory_wind_field
from dynamics.ppo_policy import PPOGuide

class TrajectoryDataset:
    def __init__(self, num_trajectories=config.NUM_TRAJECTORIES, cache_dir=config.DATA_DIR):
        self.num_trajectories = num_trajectories
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, "trajectories.npz")

        self.mock_args = type('Args', (), {
            'name': 'airship', 'action_lim': [[-10, 10], [-10, 10]],
            'max_speed': config.MAX_SPEED, 'is_action_smooth': True,
            'dt': config.DT, 'action_range': config.ACTION_RANGE,
            'pos_config': {'width': config.WIDTH, 'width_km': config.WIDTH_KM,
                           'height': config.HEIGHT, 'height_km': config.HEIGHT_KM},
            'target_range': [config.TARGET_RANGE], 'target_range_low': [15],
            'target_range_reward': [20], 'start_time': config.START_TIME,
            'start_energy': config.START_ENERGY, 'is_energy_limit': True,
            'solar_time': config.SOLAR_TIME, 'solar_power': config.SOLAR_POWER,
            'max_power': config.MAX_POWER, 'default_power': config.DEFAULT_POWER,
            'max_energy': config.MAX_ENERGY,
            'mass': 7001, 'Iz': 5.1e6, 'Iaz': 5.1e6,
            'm11': 0.11, 'm22': 0.83, 'drag_coeff': 0.045,
            'yaw_damp': 1.0, 'thrust': 1000, 'M_control': 6000,
        })()

    def _simple_action(self, airship, target, wind_s=None):
        """Controller toward target with wind compensation."""
        dx = target[0] - airship.pos_state[0]
        dy = target[1] - airship.pos_state[1]
        dist = math.sqrt(dx**2 + dy**2) + 1e-6
        vx = (dx / dist) * 0.5
        vy = (dy / dist) * 0.5
        if wind_s is not None:
            vx -= wind_s[0] * 0.3 / config.MAX_SPEED
            vy -= wind_s[1] * 0.3 / config.MAX_SPEED
        vx += random.uniform(-0.3, 0.3)
        vy += random.uniform(-0.3, 0.3)
        return np.array([vx, vy])

    def generate_trajectory(self, airship, wind_gen, wind_lists, init_pos, target_pos):
        airship.reset(init_pos, target_pos)
        wind_field_x_list, wind_field_y_list, wind_x_vector, wind_y_vector = wind_lists
        max_steps = min(config.TRAJ_LEN, len(wind_field_x_list) - 1)

        h_comp = wind_field_x_list[0].shape[1]
        w_comp = wind_field_x_list[0].shape[2]

        traj = np.zeros((max_steps, config.STATE_DIM), dtype=np.float32)
        # Initial state for PPO
        cur_as_state = np.array([
            airship.pos_state_normalize[0], airship.pos_state_normalize[1],
            airship.speed[0] / config.MAX_SPEED, airship.speed[1] / config.MAX_SPEED,
            airship.time / 24.0, airship.energy
        ], dtype=np.float32)
        for step in range(max_steps):
            wy = min(h_comp - 1, max(0, int(airship.pos_state[1] / config.RANGE_SCALE)))
            wx = min(w_comp - 1, max(0, int(airship.pos_state[0] / config.RANGE_SCALE)))
            wind_s = np.array([
                float(wind_field_x_list[step][0, wy, wx]),
                float(wind_field_y_list[step][0, wy, wx])
            ])
            wvx = wind_x_vector[step] if step < len(wind_x_vector) else np.zeros(252)
            wvy = wind_y_vector[step] if step < len(wind_y_vector) else np.zeros(252)
            action = self.ppo_guide.get_action(wvx, wvy, cur_as_state, target_pos, noise_scale=0.15)
            as_state, current_time, current_energy, current_speed_real, current_speed, wind_s, speed = \
                airship.step(action, wind_s, target_pos)
            cur_as_state = np.array([
                airship.pos_state_normalize[0], airship.pos_state_normalize[1],
                airship.speed[0] / config.MAX_SPEED, airship.speed[1] / config.MAX_SPEED,
                airship.time / 24.0, airship.energy
            ], dtype=np.float32)
            # Soft-landing at 15km: linear interpolation from entry to target
            dist_km = math.sqrt(
                ((airship.pos_state[0] - target_pos[0]) / config.WIDTH * config.WIDTH_KM) ** 2 +
                ((airship.pos_state[1] - target_pos[1]) / config.HEIGHT * config.HEIGHT_KM) ** 2
            )
            if dist_km <= 15 and step > 0:
                remaining = max_steps - step
                start_x = traj[step - 1][0]
                start_y = traj[step - 1][1]
                end_x = target_pos[0] / config.WIDTH
                end_y = target_pos[1] / config.HEIGHT
                for s in range(remaining):
                    t_val = (s + 1) / remaining
                    traj[step + s][0] = start_x + (end_x - start_x) * t_val
                    traj[step + s][1] = start_y + (end_y - start_y) * t_val
                    traj[step + s][2] = (end_x - start_x) / remaining
                    traj[step + s][3] = (end_y - start_y) / remaining
                    traj[step + s][4] = traj[step - 1][4] + (s + 1) * config.DT / 3600 / 24
                    traj[step + s][5] = traj[step - 1][5]
                break

            traj[step] = [
                airship.pos_state[0] / config.WIDTH,
                airship.pos_state[1] / config.HEIGHT,
                current_speed[0] / config.MAX_SPEED,
                current_speed[1] / config.MAX_SPEED,
                current_time / 24.0,
                current_energy,
            ]

        condition = np.array([
            init_pos[0] / config.WIDTH, init_pos[1] / config.HEIGHT,
            0.0, 0.0, config.START_TIME / 24.0, config.START_ENERGY,
            target_pos[0] / config.WIDTH, target_pos[1] / config.HEIGHT,
        ], dtype=np.float32)
        # Save first-step wind field as condition (shape: 1, H, W -> squeeze to H, W)
        # Save 4 wind frames at steps 0, 16, 32, 48
        wind_frames = [0, 8, 16, 24, 32, 40, 48, 56]
        wind_u_frames = np.stack([wind_field_x_list[s].squeeze(0).numpy().astype(np.float32) for s in wind_frames])
        wind_v_frames = np.stack([wind_field_y_list[s].squeeze(0).numpy().astype(np.float32) for s in wind_frames])
        return traj, condition, wind_u_frames, wind_v_frames

    def generate_all(self):
        wind_gen = WindGenerator(
            config.WIDTH, config.HEIGHT,
            config.WIND_MIN_AVE_V, config.WIND_MAX_AVE_V,
            config.WIND_VARY_SPEED, config.WIND_VARY_NUM,
            10, 10, config.WIND_NUM_INP, config.RANGE_SCALE
        )
        wind_buffer = Memory_wind_field(config.WIND_BUFFER_SIZE)
        for _ in range(config.WIND_BUFFER_SIZE):
            wx, wy, wvx, wvy = wind_gen.generate_wind_vector_list(config.WIND_HOURS)
            wind_buffer.add_list(wx, wy, wvx, wvy)

        airship = ASUnit_low(self.mock_args)
        self.ppo_guide = PPOGuide('./checkpoints/ppo_policy', device='cpu')
        all_trajs = []
        all_conditions = []
        all_wind_u = []
        all_wind_v = []

        for i in range(self.num_trajectories):
            if i % 500 == 0:
                print(f"Generating trajectory {i}/{self.num_trajectories}")

            wx, wy, wvx, wvy = wind_buffer.get_list()
            init_pos = [random.randint(20, config.WIDTH - 20),
                        random.randint(20, config.HEIGHT - 20)]
            target_pos = [random.randint(30, config.WIDTH - 30),
                          random.randint(30, config.HEIGHT - 30)]

            traj, cond, wu, wv = self.generate_trajectory(airship, wind_gen, (wx, wy, wvx, wvy),
                                                           init_pos, target_pos)
            all_trajs.append(traj)
            all_conditions.append(cond)
            all_wind_u.append(wu)
            all_wind_v.append(wv)

        all_trajs = np.stack(all_trajs, axis=0)
        all_conditions = np.stack(all_conditions, axis=0)
        all_wind_u = np.stack(all_wind_u, axis=0)
        all_wind_v = np.stack(all_wind_v, axis=0)
        np.savez_compressed(self.cache_file,
                           trajectories=all_trajs,
                           conditions=all_conditions,
                           wind_u=all_wind_u,
                           wind_v=all_wind_v)
        print(f"Saved {self.num_trajectories} trajectories to {self.cache_file}")
        return all_trajs, all_conditions, all_wind_u, all_wind_v

    def load(self):
        data = np.load(self.cache_file)
        return data["trajectories"], data["conditions"], data["wind_u"], data["wind_v"]

    def get_dataloader(self, batch_size=config.BATCH_SIZE, device="cpu"):
        if os.path.exists(self.cache_file):
            trajs, conds, wu, wv = self.load()
        else:
            trajs, conds, wu, wv = self.generate_all()

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(trajs, dtype=torch.float32),
            torch.tensor(conds, dtype=torch.float32),
            torch.tensor(wu, dtype=torch.float32),
            torch.tensor(wv, dtype=torch.float32)
        )
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
