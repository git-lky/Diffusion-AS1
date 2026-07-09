import numpy as np
import matplotlib.pyplot as plt
import math
import random
# from scipy.interpolate import griddata
import torch
import torch.nn.functional as F

#阵风团
class gustElement:
    def __init__(self, speed, position, radius, direction):
        self.speed = speed
        self.position = position
        self.radius = radius
        self.direction = direction  #弧度制
        # self.directionVec = [-math.sin(self.direction / 180 * math.pi), -math.cos(self.direction / 180 * math.pi)]

    # def updatePosition(self):
    #     self.position[0] -= self.magnitude * math.cos(self.direction)/baseWindSpeed
    #     self.position[1] -= self.magnitude * math.sin(self.direction)/baseWindSpeed

#风场生成器
class WindGenerator:
    def __init__(self, input_image_w, input_image_h, min_ave_v, max_ave_v, vary_speed, vary_num, vary_size,
                 vary_update_loc_span, num_inp, range_scale):
        self.x_range = input_image_w
        self.y_range = input_image_h

        self.min_ave_v = min_ave_v
        self.max_ave_v = max_ave_v
        self.vary_speed = vary_speed
        self.vary_num = vary_num
        self.vary_size = vary_size
        self.vary_update_loc_span = vary_update_loc_span    #阵风团每步移动的步长范围,值为10

        self.num_inp = num_inp  # &#25554;&#20540;&#19977;&#27425;&#65292; &#38388;&#38548;&#20026;7.5&#20998;&#38047;

        self.range_scale = range_scale

        # self._init_gusts()

    def _init_gusts(self):
        self.gusts = [gustElement(random.uniform(-self.vary_speed, self.vary_speed),
                                  [random.randint(0, self.y_range - 1), random.randint(0, self.x_range - 1)],
                                  random.randint(round(0.1 * self.y_range), round(0.3 * self.y_range)),
                                  random.randint(0, 360)) 
                                  for a in range(self.vary_num)
                    ]#生成 8 个随机阵风团，散布在地图各处

#基本风：210×120 网格上所有格点都吹同一个方向、同一个速度的风
    def _init_base_wind(self):
        self.base_wind_speed = random.uniform(self.min_ave_v, self.max_ave_v)   # 基本风速：6~10 m/s 随机取
        self.base_wind_direction = random.randint(0, 360)   # 基本风向：0~360° 随机取

    def _update_gusts_new(self):
        for gust in self.gusts:
            gust.speed += random.uniform(-1, 1)
            gust.radius += random.randint(-2, 2)
            gust.direction += random.randint(-5, 5)

            gust.speed = float(np.clip(gust.speed, -self.vary_speed, self.vary_speed))
            gust.radius = float(np.clip(
                gust.radius,
                round(0.1 * self.y_range),
                round(0.3 * self.y_range)
            ))

            if gust.direction >= 360:
                gust.direction -= 360
            if gust.direction < 0:
                gust.direction += 360

            move_step = random.randint(
                int(self.vary_update_loc_span * 0.5),
                int(self.vary_update_loc_span * 2)
            )

            gust.position[1] += move_step * math.cos(gust.direction / 180 * math.pi)
            gust.position[0] += move_step * math.sin(gust.direction / 180 * math.pi)

        new_gusts = []

        for gust in self.gusts:
            out_of_range = (
                    gust.position[0] <= 0 or
                    gust.position[0] >= (self.y_range - 1) or
                    gust.position[1] <= 0 or
                    gust.position[1] >= (self.x_range - 1)
            )

            if out_of_range:
                new_gusts.append(
                    gustElement(
                        random.uniform(
                            round(0.25 * self.base_wind_speed),
                            round(0.9 * self.base_wind_speed)
                        ),
                        [
                            random.randint(0, self.y_range - 1),
                            random.randint(0, self.x_range - 1)
                        ],
                        random.randint(
                            round(0.1 * self.y_range),
                            round(0.3 * self.y_range)
                        ),
                        random.randint(0, 360)
                    )
                )
            else:
                new_gusts.append(gust)

        self.gusts = new_gusts

    def _refresh_wind(self):
        self.base_wind_speed += random.uniform(-1, 1)
        self.base_wind_direction += random.randint(-5, 5)

        #np.clip(值, 下限, 上限) = 把值限制在下限和上限之间，超过就砍掉
        self.base_wind_speed = float(np.clip(self.base_wind_speed, self.min_ave_v, self.max_ave_v))

        if self.base_wind_direction >= 360:
            self.base_wind_direction = self.base_wind_direction - 360
        if self.base_wind_direction < 0:
            self.base_wind_direction = self.base_wind_direction + 360

    def init_wind_field(self):
        self._init_base_wind()
        self._init_gusts()

    def update_wind_field(self):
        self._refresh_wind()
        self._update_gusts_new()

    def generate_wind_field_tensor(self):
        wind_speed_array_x = self.base_wind_speed * math.cos(self.base_wind_direction / 180 * math.pi) * np.ones(
            (self.y_range, self.x_range))
        # print(wind_speed_array_x.shape)
        wind_speed_array_y = self.base_wind_speed * math.sin(self.base_wind_direction / 180 * math.pi) * np.ones(
            (self.y_range, self.x_range))

        gust_wind_field_x_list = []
        gust_wind_field_y_list = []

        for gust in self.gusts:
            gust_array = np.zeros((self.y_range, self.x_range))
            # gust_array_y = np.zeros((self.x_range, self.y_range))
            #阵风团在中心最强（权重 1.0），到边缘衰减（0.75→0.5→0.25），像一个同心圆。
            for i in range(self.y_range):
                for j in range(self.x_range):
                    dist = ((gust.position[0] - i) ** 2 + (gust.position[1] - j) ** 2) ** 0.5
                    if dist <= gust.radius:
                        gust_array[i][j] = 0.25
                        # gust_array_y[i][j] = 0.25
                    if dist <= gust.radius * 0.75:
                        gust_array[i][j] = 0.5
                        # gust_array_y[i][j] = 0.5
                    if dist <= gust.radius * 0.5:
                        gust_array[i][j] = 0.75
                        # gust_array_y[i][j] = 0.75
                    if dist <= gust.radius * 0.25:
                        gust_array[i][j] = 1
                        # gust_array_y[i][j] = 1
            gust_wind_field_x = gust_array * gust.speed * math.cos(gust.direction / 180 * math.pi)
            gust_wind_field_y = gust_array * gust.speed * math.sin(gust.direction / 180 * math.pi)

            gust_wind_field_x_list.append(gust_wind_field_x)
            gust_wind_field_y_list.append(gust_wind_field_y)

            gust_wind_field_x = np.zeros((self.y_range, self.x_range))
            gust_wind_field_y = np.zeros((self.y_range, self.x_range))

        for ii in range(len(gust_wind_field_x_list)):
            gust_wind_field_x += gust_wind_field_x_list[ii]
            gust_wind_field_y += gust_wind_field_y_list[ii]

            # gust_wind_field_x = gust_wind_field_x_list.sum()

        gust_wind_field_x = np.clip(gust_wind_field_x, -self.vary_speed, self.vary_speed)
        gust_wind_field_y = np.clip(gust_wind_field_y, -self.vary_speed, self.vary_speed)

        wind_field_x = wind_speed_array_x + gust_wind_field_x + np.random.uniform(-1, 1, (120, 210)) * 0.5
        wind_field_y = wind_speed_array_y + gust_wind_field_y + np.random.uniform(-1, 1, (120, 210)) * 0.5

        return wind_field_x, wind_field_y

    def generate_wind_field_vector(self):
        wind_speed_array_x = self.base_wind_speed * math.cos(self.base_wind_direction / 180 * math.pi) * np.ones(
            (round(self.y_range / 10), round(self.x_range / 10)))
        # print(wind_speed_array_x.shape)
        wind_speed_array_y = self.base_wind_speed * math.sin(self.base_wind_direction / 180 * math.pi) * np.ones(
            (round(self.y_range / 10), round(self.x_range / 10)))

        gust_wind_field_x_list = []
        gust_wind_field_y_list = []

        for gust in self.gusts:
            gust_array = np.zeros((round(self.y_range / 10), round(self.x_range / 10)))
            # gust_array_y = np.zeros((self.x_range, self.y_range))
            for i in range(round(self.y_range / 10)):
                for j in range(round(self.x_range / 10)):
                    try:
                        gx = float(gust.position[0])
                        gy = float(gust.position[1])
                        ii = float(i)
                        jj = float(j)

                        dist = ((gx / 10 - ii) ** 2 + (gy / 10 - jj) ** 2) ** 0.5

                    except Exception as e:
                        print("[wind field error]")
                        print("gust.position =", gust.position)
                        print("type(gust.position) =", type(gust.position))
                        print("gust.position[0] =", gust.position[0], "type =", type(gust.position[0]))
                        print("gust.position[1] =", gust.position[1], "type =", type(gust.position[1]))
                        print("i =", i, "type =", type(i))
                        print("j =", j, "type =", type(j))
                        raise e
                    dist = ((gust.position[0] / 10 - i) ** 2 + (gust.position[1] / 10 - j) ** 2) ** 0.5
                    if dist <= gust.radius / 10:
                        gust_array[i][j] = 0.25
                        # gust_array_y[i][j] = 0.25
                    if dist <= gust.radius * 0.75 / 10:
                        gust_array[i][j] = 0.5
                        # gust_array_y[i][j] = 0.5
                    if dist <= gust.radius * 0.5 / 10:
                        gust_array[i][j] = 0.75
                        # gust_array_y[i][j] = 0.75
                    if dist <= gust.radius * 0.25 / 10:
                        gust_array[i][j] = 1
                        # gust_array_y[i][j] = 1
            gust_wind_field_x = gust_array * gust.speed * math.cos(gust.direction / 180 * math.pi)
            gust_wind_field_y = gust_array * gust.speed * math.sin(gust.direction / 180 * math.pi)

            gust_wind_field_x_list.append(gust_wind_field_x)
            gust_wind_field_y_list.append(gust_wind_field_y)

            gust_wind_field_x = np.zeros((round(self.y_range / 10), round(self.x_range / 10)))
            gust_wind_field_y = np.zeros((round(self.y_range / 10), round(self.x_range / 10)))

        for ii in range(len(gust_wind_field_x_list)):
            gust_wind_field_x += gust_wind_field_x_list[ii]
            gust_wind_field_y += gust_wind_field_y_list[ii]

            # gust_wind_field_x = gust_wind_field_x_list.sum()

        gust_wind_field_x = np.clip(gust_wind_field_x, -self.vary_speed, self.vary_speed)
        gust_wind_field_y = np.clip(gust_wind_field_y, -self.vary_speed, self.vary_speed)

        wind_field_x = wind_speed_array_x + gust_wind_field_x + \
                       np.random.uniform(-1, 1, (round(self.y_range / 10), round(self.x_range / 10))) * 0.5
        wind_field_y = wind_speed_array_y + gust_wind_field_y + \
                       np.random.uniform(-1, 1, (round(self.y_range / 10), round(self.x_range / 10))) * 0.5

        return wind_field_x, wind_field_y, wind_field_x.flatten(), wind_field_y.flatten()

#生成高分辨率风场时间序列
    def generate_wind_list(self, hours):
        wind_field_x_list = []
        wind_field_y_list = []

        # 初始化基本风+阵风团
        self.gusts = []
        self.init_wind_field()

        for idx, gust in enumerate(self.gusts):
            if not isinstance(gust, gustElement):
                print("init gust error:", idx, type(gust), gust)
                raise TypeError("self.gusts init failed")

        self.init_wind_field()
        #逐小时生成
        for _ in range(hours):
            self.update_wind_field()
            wind_field_x, wind_field_y = self.generate_wind_field_tensor()
            wind_field_x = torch.from_numpy(wind_field_x).unsqueeze(0)
            wind_field_y = torch.from_numpy(wind_field_y).unsqueeze(0)
            wind_field_x_list.append(wind_field_x)
            wind_field_y_list.append(wind_field_y)
        
        #时间插值 3 轮，每轮在每两帧之间插入平均值
        for ii in range(self.num_inp):
            for i in range(0, ((len(wind_field_x_list)) * 2 - 2), 2):
                # print(i)
                temp_x = (wind_field_x_list[i] + wind_field_x_list[i + 1]) / 2
                wind_field_x_list.insert(i + 1, temp_x)
                temp_y = (wind_field_y_list[i] + wind_field_y_list[i + 1]) / 2
                wind_field_y_list.insert(i + 1, temp_y)

        return wind_field_x_list, wind_field_y_list

    def generate_wind_vector_list(self, hours):
        wind_field_x_mini_list = []
        wind_field_y_mini_list = []

        wind_x_vector_list = []
        wind_y_vector_list = []

        self.init_wind_field()
        for _ in range(hours):
            self.update_wind_field()
            wind_x_field_mini, wind_y_field_mini, wind_x_vector, wind_y_vector = self.generate_wind_field_vector()
            wind_x_field_mini = torch.from_numpy(wind_x_field_mini).unsqueeze(0)
            wind_y_field_mini = torch.from_numpy(wind_y_field_mini).unsqueeze(0)
            # wind_x_vector = torch.from_numpy(wind_x_vector).unsqueeze(0)
            # wind_y_vector = torch.from_numpy(wind_y_vector).unsqueeze(0)
            wind_x_vector_list.append(wind_x_vector)
            wind_y_vector_list.append(wind_y_vector)
            wind_field_x_mini_list.append(wind_x_field_mini)
            wind_field_y_mini_list.append(wind_y_field_mini)

        for ii in range(self.num_inp):
            for i in range(0, ((len(wind_field_x_mini_list)) * 2 - 2), 2):
                # print(i)
                temp_x = (wind_field_x_mini_list[i] + wind_field_x_mini_list[i + 1]) / 2
                wind_field_x_mini_list.insert(i + 1, temp_x)
                temp_y = (wind_field_y_mini_list[i] + wind_field_y_mini_list[i + 1]) / 2
                wind_field_y_mini_list.insert(i + 1, temp_y)

                temp_x_vec = (wind_x_vector_list[i] + wind_x_vector_list[i + 1]) / 2
                wind_x_vector_list.insert(i + 1, temp_x_vec)
                temp_y_vec = (wind_y_vector_list[i] + wind_y_vector_list[i + 1]) / 2
                wind_y_vector_list.insert(i + 1, temp_y_vec)

        return wind_field_x_mini_list, wind_field_y_mini_list, wind_x_vector_list, wind_y_vector_list

    def plot_wind_contour(self, wind_field_x, wind_field_y):
        # print(self.x_range)
        x = np.arange(0, self.x_range)
        y = np.arange(0, self.y_range)
        x_hat = np.arange(0, round(self.x_range), 10)
        y_hat = np.arange(0, round(self.y_range), 10)

        X1, Y1 = np.meshgrid(x, y)
        X2, Y2 = np.meshgrid(y_hat, x_hat)

        wind_field_syn = (wind_field_x ** 2 + wind_field_y ** 2) ** 0.5

        fig = plt.figure(figsize=[21, 12])
        plt.contourf(X1, Y1, wind_field_syn)
        plt.colorbar()
        # plt.clim(vmin=10,vmax=40)

        wind_x_tensor = torch.from_numpy(wind_field_x).unsqueeze(0).unsqueeze(0)
        wind_y_tensor = torch.from_numpy(wind_field_y).unsqueeze(0).unsqueeze(0)

        # print(wind_x_tensor.size())

        wind_x_tensor = F.interpolate(wind_x_tensor,
                                      size=(int(wind_x_tensor.size(2) / 10), int(wind_x_tensor.size(3) / 10)),
                                      mode='bilinear', align_corners=True)
        wind_y_tensor = F.interpolate(wind_y_tensor,
                                      size=(int(wind_y_tensor.size(2) / 10), int(wind_y_tensor.size(3) / 10)),
                                      mode='bilinear', align_corners=True)

        # print(wind_x_tensor[0][0].size())
        # print(wind_x_tensor[0][0].T.size())

        plt.quiver(Y2, X2, wind_x_tensor[0][0].T, wind_y_tensor[0][0].T, pivot='mid')

        plt.show(block=False)
        plt.pause(1)
        plt.close()

    def plot_wind_vector_contour(self, wind_field_x, wind_field_y):
        # print(self.x_range)
        x = np.arange(0, self.x_range, 10)
        y = np.arange(0, self.y_range, 10)
        # x_hat = np.arange(0, round(self.x_range), 10)
        # y_hat = np.arange(0, round(self.y_range), 10)

        X1, Y1 = np.meshgrid(x, y)
        # X2, Y2 = np.meshgrid(y_hat, x_hat)
        X2, Y2 = np.meshgrid(y, x)

        wind_field_syn = (wind_field_x ** 2 + wind_field_y ** 2) ** 0.5

        fig = plt.figure(figsize=[21, 12])
        plt.contourf(X1, Y1, wind_field_syn)
        plt.colorbar()
        # plt.clim(vmin=10,vmax=40)

        wind_x_tensor = torch.from_numpy(wind_field_x).unsqueeze(0).unsqueeze(0)
        wind_y_tensor = torch.from_numpy(wind_field_y).unsqueeze(0).unsqueeze(0)

        # print(wind_x_tensor.size())
        #
        # wind_x_tensor = F.interpolate(wind_x_tensor,
        #                               size=(int(wind_x_tensor.size(2) / 10), int(wind_x_tensor.size(3) / 10)),
        #                               mode='bilinear', align_corners=True)
        # wind_y_tensor = F.interpolate(wind_y_tensor,
        #                               size=(int(wind_y_tensor.size(2) / 10), int(wind_y_tensor.size(3) / 10)),
        #                               mode='bilinear', align_corners=True)

        # print(wind_x_tensor[0][0].size())
        # print(wind_x_tensor[0][0].T.size())

        plt.quiver(Y2, X2, wind_x_tensor[0][0].T, wind_y_tensor[0][0].T, pivot='mid')

        plt.show(block=False)
        plt.pause(1)
        plt.close()


