import math


def trans_pos(init_pos_lng, init_pos_lat, target_pos_lng, target_pos_lat, dx, dy, left_bottom, scale):
    # cleaned
    init_pos = [0, 0]
    init_pos[0] = (init_pos_lng - left_bottom[0]) / dx * scale
    init_pos[1] = (init_pos_lat - left_bottom[1]) / dy * scale
    target_pos = [0, 0]
    target_pos[0] = (target_pos_lng - left_bottom[0]) / dx * scale
    target_pos[1] = (target_pos_lat - left_bottom[1]) / dy * scale
    print(init_pos, target_pos)
    return init_pos, target_pos


class PosTransformer:
    def __init__(self, width, height, width_km, height_km, left=110, bottom=15) -> None:
        self.width = width  # &#20687;&#32032;&#23485;&#24230; 210
        self.height = height  # &#20687;&#32032;&#39640;&#24230; 120
        self.width_km = width_km  # &#20844;&#37324;&#23485;&#24230; 415km
        self.height_km = height_km  # &#20844;&#37324;&#39640;&#24230; 305km
        self.left = left  # &#32463;&#24230; &#24038;&#36793;&#30028;
        self.bottom = bottom  # &#32428;&#24230; &#19979;&#36793;&#30028;

        self.rad2deg = 180 / math.pi  # &#35282;&#24230;&#21333;&#20301;&#36716;&#25442;
        self.deg2rad = math.pi / 180
        self.a = 6378137.0
        self.fl = 1 / 298.257

    def _get_geodeg(self, lng, lat, dis_lng, dis_lat):
        d_lng = dis_lng / self.a * math.cos(lat * self.deg2rad) * self.rad2deg
        d_lat = dis_lat / self.a * self.rad2deg
        return d_lng, d_lat

    def _get_distance(self, pos1, pos2):
        # pos: lon, lat
        f = self.deg2rad * (pos1[1] + pos2[1]) / 2
        g = self.deg2rad * (pos1[1] - pos2[1]) / 2
        l = self.deg2rad * (pos1[0] - pos2[0]) / 2
        sg = math.sin(g)
        sl = math.sin(l)
        sf = math.sin(f)
        sg = sg * sg
        sl = sl * sl
        sf = sf * sf
        s = sg * (1 - sl) + (1 - sf) * sl
        c = (1 - sg) * (1 - sl) + sf * sl
        w = math.atan(math.sqrt(s / c))
        r = math.sqrt(s * c) / w
        d = 2 * w * self.a
        h1 = (3 * r - 1) / 2 / c
        h2 = (3 * r + 1) / 2 / s
        s = d * (1 + self.fl * (h1 * sf * (1 - sg) - h2 * (1 - sf) * sg)) / 1000
        return s

    def get_pos_gps(self, pos):
        # cleaned
        x = pos[0]
        y = pos[1]
        dis_x = x / self.width * self.width_km
        dis_y = y / self.height * self.height_km

        d_lng, d_lat = self._get_geodeg(self.left, self.bottom, dis_x, dis_y)
        return self.left + d_lng, self.bottom + d_lat
