import os
import time
import math
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import random as rd
from matplotlib.gridspec import GridSpec
import torch



class ASUnit_low(object):
    def __init__(self, args):
        self.ID = args.name
        self.action = np.zeros(len(args.action_lim))
        self.max_speed = args.max_speed
        self.action_real = []
        self.is_action_smooth = args.is_action_smooth
        self.dt = args.dt  # 3600x 7.5*60
        self.steps = 0
        self.init_pos = [0, 0]
        self.pos_state = np.array(self.init_pos, dtype="float")
        self._pos_state = self.pos_state
        self.pos_state_ep = np.expand_dims(self.pos_state, 1)

        self.action_lim = args.action_lim

        self.speed = np.zeros(len(self.action_lim))
        self.speed_ep = np.expand_dims(self.speed, 1)
        self.action_range = args.action_range
        self.speed_range = [-self.max_speed, self.max_speed]

        # self.pos_query = np.expand_dims(np.array(as_p.init_pos[:], dtype ='float'), 1)  # 状态向量
        # self.action_query = np.array([[] for _ in range(len(self.action_lim))])  # 动作向量
        self.time_flag = False

        self.rad2deg = 180 / math.pi  # 角度单位转换
        self.deg2rad = math.pi / 180
        self.a = 6378137.0
        self.fl = 1 / 298.257

        self.pos_config = args.pos_config  # width_km: 幅宽, height_km: 幅高, width: 像素宽, height: 像素高

        # reaward part

        # self.target_pos = as_p.target_pos
        # self.avoid_ranges = args.avoid_ranges
        self.target_range = args.target_range_low  # [375, 750]
        self.target_range_reward = args.target_range_reward

        self.energy_k = 0.1  # 能源消耗系数
        self.time = args.start_time
        self.start_time = args.start_time
        self.energy = args.start_energy
        self.start_energy = args.start_energy
        self.is_energy_limit = args.is_energy_limit

        self.solar_time = args.solar_time
        self.solar_power = args.solar_power
        self.max_power = args.max_power
        self.default_power = args.default_power
        self.max_energy = args.max_energy

        self.solars = [[5, 13, 17], [0, 18, 0]]     # 含义：早上 5 点太阳能=0，下午 1 点=18kW 峰值，傍晚 5 点=0
        self.solars_x = np.array(self.solars[0])    # 对这三个点做二次多项式拟合，得到一条抛物线，用来估算任意时刻的太阳能
        self.solars_y = np.array(self.solars[1])
        self.coef = np.polyfit(self.solars_x, self.solars_y, 2)  # 最小二乘法进行多项式拟合
        self.angle = rd.uniform(0, 360)

        self.mass = args.mass
        self.Iz = args.Iz  # from your doc example (kg·m^2)
        self.Iaz = args.Iaz  # added rotational inertia (rough)
        self.m11 = args.m11
        self.m22 = args.m22
        self.Ma = np.array([self.m11 * self.mass, self.m22 * self.mass])  # from your file's k11,k22
        self.drag_coeff = args.drag_coeff
        self.yaw_damp = args.yaw_damp
        self.thrust = args.thrust
        self.M_control = args.M_control
        self.dt = args.dt  # seconds (ensure correct)

    def reset(self, init_pos_, target_pos):
        # init_pos: 像素
        # target_pos: 像素
        # print("reset init pos:",init_pos_)
        self.taeget_final = target_pos
        self.target_pos = target_pos
        self.pos_state = np.array(init_pos_[:], dtype='float')
        self._pos_state = self.pos_state
        self.pos_state_normalize = np.array(self._convert_query(init_pos_), dtype="float")
        self.pos_query = np.expand_dims(self.pos_state_normalize, 1)
        self.pos_state_ep = np.expand_dims(self.pos_state, 1)

        self.action = np.zeros(len(self.action_lim))
        self.speed = np.zeros(len(self.action_lim))
        self.speed_ep = np.expand_dims(self.speed, 1)
        self.action_real = []
        self.action_query = np.array([[] for _ in range(len(self.action_lim))])
        self.steps = 0
        as_state = np.concatenate((self.pos_state_normalize, self.speed / self.max_speed), 0)
        if self.is_energy_limit:
            self.energy = self.start_energy
            self.time = self.start_time
            as_state = np.concatenate((as_state, np.array([self.time / 24, self.energy])), 0)
        return as_state

    # --------- 辅助方法：双线性插值----------
    def interp_wind_and_grad(self, wind_field_x, wind_field_y, pos):
        """
        双线性插值获取风速 + 有限差分获取梯度
        不考虑旋度,rw = grad_rw = 0
        """

        # -------- 1. 保证是 numpy 2D --------
        def to_numpy(w):
            if torch.is_tensor(w):
                w = w.squeeze().detach().cpu().numpy()
            else:
                w = np.array(w)
            assert w.ndim == 2, f"wind field must be 2D, got {w.shape}"
            return w

        wx = to_numpy(wind_field_x)
        wy = to_numpy(wind_field_y)

        H, W = wx.shape

        # -------- 2. 连续坐标（缩放）--------
        x = pos[0]
        y = pos[1]

        # clamp 到网格内部（避免越界）
        x = np.clip(x, 0, W - 1.001)
        y = np.clip(y, 0, H - 1.001)

        # -------- 3. 双线性插值 --------
        x0 = int(np.floor(x))
        x1 = x0 + 1
        y0 = int(np.floor(y))
        y1 = y0 + 1

        x1 = min(x1, W - 1)
        y1 = min(y1, H - 1)

        dx = x - x0
        dy = y - y0

        def bilinear(M):
            return (
                    M[y0, x0] * (1 - dx) * (1 - dy) +
                    M[y0, x1] * dx * (1 - dy) +
                    M[y1, x0] * (1 - dx) * dy +
                    M[y1, x1] * dx * dy
            )

        Vx = bilinear(wx)
        Vy = bilinear(wy)
        Vw = np.array([Vx, Vy], dtype=np.float32)

        # -------- 4. 局部有限差分梯度（基于网格）--------
        # 这里仍旧使用网格差分，因为风场是基于网格生成的
        xg = int(np.clip(round(x), 1, W - 2))
        yg = int(np.clip(round(y), 1, H - 2))

        dVx_dx = (wx[yg, xg + 1] - wx[yg, xg - 1]) * 0.5
        dVx_dy = (wx[yg + 1, xg] - wx[yg - 1, xg]) * 0.5

        dVy_dx = (wy[yg, xg + 1] - wy[yg, xg - 1]) * 0.5
        dVy_dy = (wy[yg + 1, xg] - wy[yg - 1, xg]) * 0.5

        gradV = np.array([
            [dVx_dx, dVx_dy],
            [dVy_dx, dVy_dy]
        ], dtype=np.float32)

        # -------- 5. 不考虑涡旋（按你要求）--------
        rw = 0.0
        grad_rw = np.zeros(2, dtype=np.float32)

        return Vw, gradV, rw, grad_rw

    def step_low(self, action, wind_field_s, target_pos, ep=None):
        """
        action: tensor/np like shape (2,) -> [T, Mz]
        wind_field_tuple: (wind_field_x, wind_field_y) np arrays in pixel grid
        returns: (as_state, current_time, current_energy, current_speed_real, current_speed, wind_s, speed_ground)
        """
        # unpack action
        # ensure numpy array
        act = np.array(action).astype(float).squeeze()
        T = float(act[0]) * self.thrust # main thrust (N)
        Mz = float(act[1]) * self.M_control  # yaw torque (N*m)

        wind_field_x, wind_field_y = wind_field_s

        # --- parameters (make sure these exist in __init__ or set defaults) ---
        # required attributes (give defaults if missing)

        # self.speed expected shape (2,) = [u, v] (机体或地速？我们取机体速度 relative to air)
        # self.pos_state expected [x_pix, y_pix]
        # self.r (yaw rate), self.psi (yaw angle) expected

        # --- RK4 state: [u, v, psi, r, x_pix, y_pix] ---
        def pack_state():
            return np.array([self.speed[0], self.speed[1], getattr(self, "psi", 0.0), getattr(self, "r", 0.0),
                             self.pos_state[0], self.pos_state[1]], dtype=float)

        def unpack_state(s):
            u, v, psi, r, xpix, ypix = s
            return u, v, psi, r, np.array([xpix, ypix])

        # dynamics function uses local pos and speed to compute derivatives
        def dynamics(s_vec):
            u, v, psi, r, pos = *s_vec[:2], s_vec[2], s_vec[3], np.array([s_vec[4], s_vec[5]])
            # note: above line python-unpacking style; ensure consistent
            # safer:
            u = float(s_vec[0]); # x轴方向速度，空速
            v = float(s_vec[1]); # y轴方向速度，空速
            psi = float(s_vec[2]); # 偏航角
            r = float(s_vec[3]) # 偏航角速度
            pos = np.array([float(s_vec[4]), float(s_vec[5])])

            # --- interpolate wind and gradient at pos ---
            Vw, gradV, rw, grad_rw = self.interp_wind_and_grad(wind_field_x, wind_field_y, pos)

            # --- compute convective derivative of Vw: (V_ground · grad) Vw
            # Choose V_ground = u + Vw? In Zhao formulation, V is ground speed; approximate ground speed = body speed + Vw
            V_air = np.array([u, v])  # body-frame velocities (assumed)
            V_ground = (V_air+ Vw)
            # compute (V · ∇) Vw = [ u*∂xVx + v*∂yVx; u*∂xVy + v*∂yVy ]
            # gradV rows: [dVx/dx, dVx/dy]; [dVy/dx, dVy/dy]
            dotVw = np.array([
                V_ground[0] * gradV[0, 0] + V_ground[1] * gradV[0, 1],
                V_ground[0] * gradV[1, 0] + V_ground[1] * gradV[1, 1]
            ])

            # --- wind angular convective derivative
            rw_dot = V_ground[0] * grad_rw[0] + V_ground[1] * grad_rw[1]  # grad_rw approximated earlier (likely zero)

            # --- wind-induced force using Zhao model ---
            # Fw = Ma * dotVw + omega x (Ma * Vw)
            Ma = np.array(self.Ma)  # diag added-mass [ma_x, ma_y]
            MaVw = Ma * Vw
            F_added = Ma * dotVw
            # omega x (MaVw) for planar: omega = [0,0,r], cross gives [-r*MaVw_y, r*MaVw_x]
            F_coriolis = np.array([-r * MaVw[1], r * MaVw[0]])
            Fw = F_added + F_coriolis

            # --- aerodynamic drag (simplified quadratic) ---
            V_rel = V_air - Vw  # body velocity relative to air
            Vrel_norm = np.linalg.norm(V_rel) + 1e-6
            Fd = - self.drag_coeff * V_rel * Vrel_norm

            # --- total force in body-frame (Tx applied along body x, Ty neglected because action only T along x) ---
            # T is thrust magnitude along body x; assume small lateral thrust from control surfaces neglected
            # If you want lateral thrust, change action dimension
            # Here action T behaves as axial thrust generating acceleration along u (body x)
            Fx = T
            Fy = 0.0

            F_total = np.array([Fx, Fy]) + Fw + Fd

            # --- accelerate (divide by (m + Ma) if you want to include added mass effect on inertia) ---
            # we use effective mass = m + Ma (elementwise)
            m_eff_x = self.mass + Ma[0]
            m_eff_y = self.mass + Ma[1]
            a_u = F_total[0] / m_eff_x
            a_v = F_total[1] / m_eff_y

            # --- yaw/rotational dynamics: I_eff * r_dot = Mz + M_w - yaw_damp * r
            # Mw approx Iaz * rw_dot + small coupling: + r * Iaz * rw (can be added)
            I_eff = self.Iz + getattr(self, "Iaz", 0.0)
            Mw = getattr(self, "Iaz", 0.0) * rw_dot + 0.0  # simplification
            r_dot_local = (Mz + Mw - self.yaw_damp * r) / I_eff

            # --- kinematics: pos_dot in pixel coordinates: convert ground speed (m/s) -> pixel delta per sec
            # First compute ground velocity in meters/sec: V_ground = V_body + Vw (both m/s)
            # Convert meters to pixel by same mapping you used earlier:
            Vg = V_ground  # [m/s, m/s]
            # Convert to pixel per second:
            d_lng_m = Vg[0]  # m/s along x
            d_lat_m = Vg[1]  # m/s along y
            # use pos_config mapping (width_km,height_km -> width,height pixels)
            pix_dx = d_lng_m / 1000.0 / self.pos_config["width_km"] * self.pos_config["width"]
            pix_dy = d_lat_m / 1000.0 / self.pos_config["height_km"] * self.pos_config["height"]

            # state derivative vector
            # u_dot, v_dot, psi_dot (= r), r_dot, x_dot_pix, y_dot_pix
            return np.array([a_u, a_v, r, r_dot_local, pix_dx, pix_dy], dtype=float)

        # --- RK4 integration over dt seconds ---
        s0 = pack_state()
        dt = float(self.dt)

        k1 = dynamics(s0)
        k2 = dynamics(s0 + 0.5 * dt * k1)
        k3 = dynamics(s0 + 0.5 * dt * k2)
        k4 = dynamics(s0 + dt * k3)

        s_next = s0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # unpack and assign back to env
        u_next, v_next, psi_next, r_next = s_next[0], s_next[1], s_next[2], s_next[3]
        pos_next = np.array([s_next[4], s_next[5]])

        self.speed[0] = u_next
        self.speed[1] = v_next
        self.psi = psi_next
        self.r = r_next
        self.pos_state = pos_next

        # after step energy/time updates (keep your original)
        self.time = self.time + self.dt / 3600.0
        if self.time > 24:
            self.time -= 24.0

        # recalc wind at new pos for outputs
        Vw_new, _, _, _ = self.interp_wind_and_grad(wind_field_x, wind_field_y, self.pos_state)
        speed_ground = self.speed + Vw_new
        self.real_speed = np.linalg.norm(speed_ground)

        # call your energy step function (keep original behavior)
        self.step_energy(self.real_speed, self.time)

        # clamp speeds as in your original code (keep same limits)
        for i in range(len(self.action_lim)):
            if self.speed[i] > self.speed_range[1]:
                self.speed[i] = self.speed_range[1]
            if self.speed[i] < self.speed_range[0]:
                self.speed[i] = self.speed_range[0]

        if self.real_speed > self.speed_range[1]:
            self.speed = self.speed * (self.max_speed / self.real_speed)
            self.real_speed = self.speed_range[1]

        # construct observation similar to your original
        self.pos_state_normalize = np.array(self._convert_query(self.pos_state), dtype=float)
        as_state = np.concatenate((self.pos_state_normalize, self.speed / self.max_speed), axis=0)
        if self.is_energy_limit:
            as_state = np.concatenate((as_state, np.array([self.time / 24.0, self.energy])), axis=0)

        # return same signature as your original step
        current_energy = self.energy
        current_time = self.time
        current_speed_real = self.real_speed
        current_speed = self.speed.copy()
        wind_s = Vw_new
        speed = speed_ground

        return as_state, current_time, current_energy, current_speed_real, current_speed, wind_s, speed
        # return as_state, current_time, current_energy, current_speed_real, current_speed, wind_s, speed

    def solar_energy(self, time):
        energy = 0
        if time > self.solar_time[0] and time < self.solar_time[1]:
            energy = np.polyval(self.coef, time) * 1.2  # 用拟合的抛物线算，再放大 1.2 倍
        return energy

    def step_energy(self, speed, time):
        consumption_energy = (speed / self.max_speed) ** 3 * self.max_power + self.default_power
        solar_energy = self.solar_energy(time)
        self.energy = self.energy + (solar_energy - consumption_energy) * self.dt / 3600 / self.max_energy
        if self.energy > 1:
            self.energy = 1
        if self.energy < 0:
            self.energy = 0
        # print("self.energy=", self.energy)
        return self.energy

    # 方法3：最安全的版本，确保所有情况都处理
    def safe_to_numpy(data):
        """安全转换为numpy数组"""
        if data is None:
            return None

        # PyTorch张量
        if hasattr(data, 'is_cuda') and hasattr(data, 'detach'):
            if data.is_cuda:
                return data.detach().cpu().numpy()
            else:
                return data.detach().numpy()

        # NumPy数组
        elif isinstance(data, np.ndarray):
            return data

        # 其他类型
        else:
            try:
                return np.array(data)
            except:
                return data



    def step(self, action, wind_s, target_pos, ep=None):
        # action： 经度速度，纬度速度 m/s
        self.target_pos = target_pos
        self.action = action
        self.speed_before = self.speed

        # 使用方式
        if isinstance(self.action_range, np.ndarray):
            self.action_range = torch.tensor(self.action_range, dtype=torch.float32, device=self.action.device)

        self.action_real = self.action_range * self.action
        self.speed += self.action_real.squeeze()
        # step for time
        self.time = self.time + self.dt / 3600
        if self.time > 24:
            self.time = self.time - 24

        for i in range(len(self.action_lim)):
            if self.speed[i] > self.speed_range[1]:
                self.speed[i] = self.speed_range[1]
            if self.speed[i] < self.speed_range[0]:
                self.speed[i] = self.speed_range[0]

        self.real_speed = np.sqrt(self.speed[0] ** 2 + self.speed[1] ** 2)
        if self.real_speed > self.speed_range[1]:
            self.speed[0] = self.speed[0] * (self.max_speed / self.real_speed)
            self.speed[1] = self.speed[1] * (self.max_speed / self.real_speed)
            self.real_speed = self.speed_range[1]

        # step for energy
        self.energy = self.step_energy(self.real_speed, self.time)
        if self.energy < 0.1:
            for i in range(len(self.action_lim)):
                self.speed[i] = 0
                self.real_speed = 0
        # according to energy modify the speed
        speed = self.speed + wind_s
        dis_lng = self.dt * speed[0]  # m
        dis_lat = self.dt * speed[1]  # m

        d_lng = dis_lng / 1000 / self.pos_config["width_km"] * self.pos_config["width"]
        d_lat = dis_lat / 1000 / self.pos_config["height_km"] * self.pos_config["height"]

        self._pos_state = self._temp(self.pos_state)  # _pos_state 前一步状态

        self.pos_state[0] = self.pos_state[0] + d_lng  # 像素宽
        self.pos_state[1] = self.pos_state[1] + d_lat  # 像素高

        self.pos_state_normalize = np.array(self._convert_query(self.pos_state), dtype="float")
        self.pos_state_ep = np.append(self.pos_state_ep, np.expand_dims(self.pos_state, 1), axis=1)
        self.speed_ep = np.append(self.speed_ep, np.expand_dims(self.speed, 1), axis=1)
        self.pos_query = np.append(self.pos_query, np.expand_dims(self.pos_state_normalize, 1), axis=1)
        as_state = np.concatenate((self.pos_state_normalize, self.speed / self.max_speed), 0)
        if self.is_energy_limit:
            as_state = np.concatenate((as_state, np.array([self.time / 24, self.energy])), 0)
        # print("as states:",self.pos_state_normalize, speed, self.speed, wind_s, action)
        current_energy = self.energy
        current_time = self.time
        current_speed_real = self.real_speed
        # print("current_speed_real", current_speed_real)
        current_speed = self.speed
        # print("current speed",current_speed)
        return as_state, current_time, current_energy, current_speed_real, current_speed, wind_s, speed

    def _temp(self, pos_state):
        return pos_state.copy()

    def _convert_query(self, pos_state):
        # 归一化处理，pos_state 像素单位
        lng_rate = pos_state[0] / self.pos_config["width"]
        lat_rate = pos_state[1] / self.pos_config["height"]
        return [lng_rate, lat_rate]


    def _convert_geo(self, pos_query):
        lng_q = pos_query[0]
        lat_q = pos_query[1]

        return [lng_q * (self.pos_config["width"][1] - self.pos_config["width"][0]) + self.pos_config["width"][0],
                lat_q * (self.pos_config["height"][1] - self.pos_config["height"][0]) + self.pos_config["height"][0]]

    def _get_geodeg(self, lng, lat, dis_lng, dis_lat):
        d_lng = dis_lng / self.a * math.cos(lat * self.deg2rad) * self.rad2deg
        d_lat = dis_lat / self.a * self.rad2deg
        return d_lng, d_lat

    def _get_distance(self, pos1, pos2):
        # pos: 像素 宽, 像素 高
        dis = pos1 - pos2
        dis = [
            dis[0] / self.pos_config["width"] * self.pos_config["width_km"],
            dis[1] / self.pos_config["height"] * self.pos_config["height_km"]]
        s = math.sqrt(dis[0] ** 2 + dis[1] ** 2)

        return s

    def reward_energy(self):

        consumption = self.energy_k * (self.speed[0] ** 2 + self.speed[1] ** 2) ** 1.5
        # print("consumption=", consumption)

        return consumption

    def reward_distance(self):
        dis = self._get_distance(self.pos_state, self.target_pos)
        _dis = self._get_distance(self._pos_state, self.target_pos)

        r = (_dis - dis) / 5

        return r

    def reward_yaw(self):
        # 前一时刻指向目标的单位向量
        self.speed_before = self.speed_before / (np.linalg.norm(self.speed_before)+1e-8)
        vec_before = self.target_pos - self._pos_state
        vec_before = vec_before / (np.linalg.norm(vec_before) + 1e-8)
        cos_before = np.dot(self.speed_before, vec_before)

        # 后一时刻指向目标的单位向量
        self.speed_normalize = self.speed / (np.linalg.norm(self.speed) + 1e-8)
        vec_after = self.target_pos - self.pos_state
        vec_after = vec_after / (np.linalg.norm(vec_after) + 1e-8)
        cos_after = np.dot(self.speed_normalize, vec_after)
        r = cos_after - cos_before
        r = 0.5 * r
        return r

    # def reward(self):
    #     reward_energy = -0.0003 * self.reward_energy()
    #     dr = self.reward_distance()
    #     step_reward = -0.01
    #
    #     mission_reward = 0
    #     if self.energy < 0.1:
    #         mission_reward -= 0.2
    #
    #     return dr + mission_reward + step_reward + reward_energy

    # def reward_test(self):
    #     reward_energy =  - 0.0003 * self.reward_energy()
    #     print("reward_energy:", reward_energy)
    #     dr = self.reward_distance()
    #     step_reward = -0.01
    #     print("reward distance:", dr)
    #     # print("step_reward:", step_reward)
    #
    #     mission_reward = 0
    #
    #     if self.energy < 0.1:
    #         mission_reward -= 0.2
    #     # print("mission reward:", mission_reward)
    #
    #     return dr + mission_reward + step_reward + reward_energy
    def reward(self, actions=None):
        dr_raw = self.reward_distance()

        # 限制距离奖励，不让单步靠近奖励过大
        dr_clip = np.clip(dr_raw, -0.2, 0.2)
        reward_distance = 0.2 * dr_clip  # 范围约 [-0.04, +0.04]

        step_reward = -0.01

        reward_energy = -0.0001 * self.reward_energy()

        mission_reward = 0.0
        if self.energy < 0.1:
            mission_reward -= 0.1

        action_penalty = 0.0
        if actions is not None:
            if isinstance(actions, torch.Tensor):
                action_np = actions.detach().cpu().numpy()
            else:
                action_np = np.asarray(actions)

            action_norm = np.mean(np.square(action_np))
            action_penalty = -0.002 * action_norm

        reward = reward_distance + step_reward + reward_energy + mission_reward + action_penalty

        return reward

    def reward_test(self, actions=None):
        dr_raw = self.reward_distance()

        # 限制距离奖励，不让单步靠近奖励过大
        dr_clip = np.clip(dr_raw, -0.2, 0.2)
        reward_distance = 0.2 * dr_clip  # 范围约 [-0.04, +0.04]
        print("reward distance=", reward_distance)

        step_reward = -0.01

        reward_energy = -0.0001 * self.reward_energy()
        print("reward energy=", reward_energy)

        mission_reward = 0.0
        if self.energy < 0.1:
            mission_reward -= 0.1

        action_penalty = 0.0
        if actions is not None:
            if isinstance(actions, torch.Tensor):
                action_np = actions.detach().cpu().numpy()
            else:
                action_np = np.asarray(actions)

            action_norm = np.mean(np.square(action_np))
            action_penalty = -0.002 * action_norm
            print("action penalty=", action_penalty)

        reward = reward_distance + step_reward + reward_energy + mission_reward + action_penalty

        return reward

    def get_time(self, ):
        """
        获取当前日期和时间
        :return: none
        """
        self.date = time.strftime('%Y-%m-%d', time.localtime())
        self.now_time = time.strftime('%H_%M_%S', time.localtime())

    def save_data(self, ep_reward=None, ep=None, env_state=None):
        """
        保存状态、动作序列为.npy文件，命名规则为当前时间
        :return: none
        """
        print("当前项目路径:" + os.getcwd())
        self.get_time()
        if self.time_flag is False: self.time_flag = True
        if ep_reward:
            file_path = os.getcwd() + "/" + self.date + self.now_time + "/ep_" + str(ep) + "/env_data/"
        else:
            file_path = os.getcwd() + "/" + self.date + self.now_time + "/ep_" + str(ep) + "/" + self.ID + "/s_a_r/"

        try:
            if not os.path.exists(file_path):
                os.makedirs(file_path)
                print("数据目录创建成功:" + file_path)
        except BaseException as msg:
            print("创建数据目录失败:" + str(msg))

        if ep_reward:
            np.save(file_path + self.ID + "episode_reward.npy", ep_reward)
            if env_state:
                np.save(file_path + "env_state.npy", env_state)
        else:
            np.save(file_path + self.ID + "_state.npy", self.pos_state_ep)
            np.save(file_path + self.ID + "_action.npy", self.speed_ep)

    def plot_state(self, ep=None):
        """
        绘制模型状态并保存
        :return: none
        """
        # 设置fig属性
        font = {'size': 16, 'family': 'serif'}
        fig_size = [10, 20] if self.is_single else [10, 10]
        figure = {'figsize': fig_size, 'dpi': 100.0}
        plt.rc('font', **font)
        plt.rc('figure', **figure)
        fig = plt.figure(constrained_layout=True)
        GS = GridSpec(4, 1, fig) if self.is_single else GridSpec(1, 1, fig)
        # 绘制起点、终点、路径
        ax_0 = fig.add_subplot(GS[0: 2, :]) if self.is_single else GS[0, 0]
        ax_0.plot(self.pos_query[0], self.pos_query[1], marker='o', color='C0', ms=2, lw=0.5)  # 路径
        ax_0.plot(self.pos_query[0][0], self.pos_query[1][0], marker='*', color='darkblue', ms=16)  # 起点
        ax_0.plot(self.pos_query[0][-1], self.pos_query[1][-1], marker='*', color='C3', ms=16)  # 终点
        # 设置axe属性
        lim = [0, 1]
        ax_0.set_xlim(lim)
        ax_0.set_ylim(lim)
        ax_0.set_xlabel('X')
        ax_0.set_ylabel('Y')
        ax_0.grid()
        # 保存图像
        if self.time_flag is False:
            self.get_time()
        else:
            self.time_flag = False
        file_path = os.getcwd() + "/" + self.date + self.now_time + "/ep_" + str(ep) + "/" + self.ID + "/fig/"
        try:
            if not os.path.exists(file_path):
                os.makedirs(file_path)
                print("图片目录创建成功：" + file_path)
        except BaseException as msg:
            print("创建图片目录失败：" + str(msg))
        plt.savefig(file_path + self.ID + "_state_plot.png")
        plt.show()









